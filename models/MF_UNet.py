import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import time
from PIL import Image


class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.act1 = nn.LeakyReLU(2e-1, True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    return x

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1


class DepthConv(nn.Module):
    def __init__(self, fmiddle, kw=3, padding=1, stride=1):
        super().__init__()

        self.kw = kw
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(self.kw, self.kw), dilation=1, padding=1, stride=stride)
        BNFunc = nn.BatchNorm2d
        self.norm_layer = BNFunc(fmiddle, affine=True)

    def forward(self, x, conv_weights):
        N, C, H, W = x.size()
        conv_weights = conv_weights.view(N * C, self.kw * self.kw, H // self.stride, W // self.stride)
        # conv_weights = nn.functional.softmax(conv_weights, dim=1)
        x = self.unfold(x).view(N * C, self.kw * self.kw, H // self.stride, W // self.stride)
        x = torch.mul(conv_weights, x).sum(dim=1, keepdim=False).view(N, C, H // self.stride, W // self.stride)
        # x = self.norm_layer(x)
        return x


class DepthsepCCBlock(nn.Module):
    def __init__(self, fin, fout, feature_dim):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # layers to generate conditional convolution weights
        nhidden = 128
        self.weight_channels = fmiddle * 9
        self.gen_weights1 = nn.Sequential(
            nn.Conv2d(feature_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fin * 9, kernel_size=3, padding=1))
        self.gen_weights2 = nn.Sequential(
            nn.Conv2d(feature_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout * 9, kernel_size=3, padding=1))

        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(feature_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fmiddle, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.gen_se_weights2 = nn.Sequential(
            nn.Conv2d(feature_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout, kernel_size=3, padding=1),
            nn.Sigmoid())

        # create conv layers
        BNFunc = nn.BatchNorm2d
        self.conv_0 = DepthConv(fin)
        self.norm_0 = BNFunc(fmiddle, affine=True)
        self.conv_1 = nn.Conv2d(fin, fmiddle, kernel_size=1)
        self.norm_1 = BNFunc(fin, affine=True)
        self.conv_2 = DepthConv(fmiddle)
        self.norm_2 = BNFunc(fmiddle, affine=True)
        self.conv_3 = nn.Conv2d(fmiddle, fout, kernel_size=1)

        self.learned_shortcut = (fin != fout)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.norm_s = nn.BatchNorm2d(fin)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def forward(self, trad_mask, img_feature):

        # predict weight for conditional convolution
        segmap = F.interpolate(img_feature, size=trad_mask.size()[2:], mode='nearest')
        conv_weights1 = self.gen_weights1(segmap)
        conv_weights2 = self.gen_weights2(segmap)
        se_weights1 = self.gen_se_weights1(segmap)
        se_weights2 = self.gen_se_weights2(segmap)

        mask_s = self.shortcut(trad_mask)

        dx = self.norm_1(trad_mask)
        dx = self.conv_0(dx, conv_weights1)
        dx = self.conv_1(dx)
        dx = torch.mul(dx, se_weights1)
        dx = self.actvn(dx)

        dx = self.norm_2(dx)
        dx = self.conv_2(dx, conv_weights2)
        dx = self.conv_3(dx)
        dx = torch.mul(dx, se_weights2)
        dx = self.actvn(dx)

        out = mask_s + dx

        return out


# This model shrinks the input traditional mask to 1/16 of the original resolution, i.e. 16*16,
# which might be too small, causing the model to output nothing like the input.
class MF_UNet(nn.Module):
    def __init__(self, n_class, sigma=1, YLength=10):
        super().__init__()
        # For Match Filtering, which is the traditional method
        self.filters = self.get_filters_as_conv(sigma, YLength)
        self.filters.requires_grad_(False)

        # For extracting image information:
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())

        self.img_enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.img_enc2 = nn.Sequential(*self.base_layers[3:5])
        self.img_enc3 = self.base_layers[5]
        self.img_enc4 = self.base_layers[6]
        self.img_enc5 = self.base_layers[7]

        self.img_dec4 = Decoder(512, 256 + 256, 256)
        self.img_dec3 = Decoder(256, 256 + 128, 256)
        self.img_dec2 = Decoder(256, 128 + 64, 128)
        self.img_dec1 = Decoder(128, 64 + 64, 64)
        self.img_dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )

        # For traditional mask processing
        # Denoising Encoder
        self.mask_enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(2e-1, True))
        self.mask_enc2 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.mask_enc3 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
        self.mask_enc4 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))
        self.mask_enc5 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))

        # Mending Decoder
        self.mask_dec5 = DepthsepCCBlock(512, 256, 256)
        self.mask_dec4 = DepthsepCCBlock(256, 128, 256)
        self.mask_dec3 = DepthsepCCBlock(128, 64, 128)
        self.mask_dec2 = DepthsepCCBlock(64, 32, 64)
        self.mask_dec1 = DepthsepCCBlock(32, 16, 64)
        self.up = nn.Upsample(scale_factor=2)

        self.conv_img = nn.Conv2d(16, 3, 3, padding=1)

    def get_matched_filtering_filters(self, sigma=1, YLength=10):
        filters = []
        widthOfTheKernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
        if np.mod(widthOfTheKernel, 2) == 0:
            widthOfTheKernel = widthOfTheKernel + 1
        widthOfTheKernel = int(widthOfTheKernel)
        # print(widthOfTheKernel)
        for theta in np.arange(0, np.pi, np.pi / 16):
            # theta = np.pi/4
            matchFilterKernel = np.zeros((widthOfTheKernel, widthOfTheKernel), dtype=np.float)
            for x in range(widthOfTheKernel):
                for y in range(widthOfTheKernel):
                    halfLength = (widthOfTheKernel - 1) / 2
                    x_ = (x - halfLength) * np.cos(theta) + (y - halfLength) * np.sin(theta)
                    y_ = -(x - halfLength) * np.sin(theta) + (y - halfLength) * np.cos(theta)
                    if abs(x_) > 3 * np.ceil(sigma):
                        matchFilterKernel[x][y] = 0
                    elif abs(y_) > (YLength - 1) / 2:
                        matchFilterKernel[x][y] = 0
                    else:
                        matchFilterKernel[x][y] = -np.exp(-.5 * (x_ / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
            m = 0.0
            for i in range(matchFilterKernel.shape[0]):
                for j in range(matchFilterKernel.shape[1]):
                    if matchFilterKernel[i][j] < 0:
                        m = m + 1
            mean = np.sum(matchFilterKernel) / m
            for i in range(matchFilterKernel.shape[0]):
                for j in range(matchFilterKernel.shape[1]):
                    if matchFilterKernel[i][j] < 0:
                        matchFilterKernel[i][j] = matchFilterKernel[i][j] - mean

            matchFilterKernel = torch.from_numpy(matchFilterKernel)
            filters.append(matchFilterKernel)

        filters = torch.stack(filters).unsqueeze(1)

        return filters

    def get_filters_as_conv(self, sigma, YLength):
        filters = self.get_matched_filtering_filters(sigma, YLength)
        widthOfTheKernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
        if np.mod(widthOfTheKernel, 2) == 0:
            widthOfTheKernel = widthOfTheKernel + 1
        widthOfTheKernel = np.uint8(widthOfTheKernel)
        pad =  np.uint8((widthOfTheKernel - 1)/2)
        conv = nn.Conv2d(1,16,kernel_size=(widthOfTheKernel, widthOfTheKernel),padding=(pad, pad))
        conv.weight.data = filters.float()
        return conv

    def get_one_hot(self, label, N=2):
        label = label.squeeze(1)
        size = list(label.size())
        label = label.view(-1)   # reshape ?????????
        ones = torch.sparse.torch.eye(N).to('cuda')
        ones = ones.index_select(0, label)   # ???????????????????????????one hot
        size.append(N)  # ?????????????????????size??????????????????reshape??????????????????
        return ones.view(*size)

    def binarize(self, src, thresh):
        temp = ((src >= thresh) * 1)
        trad_mask_res = self.get_one_hot(temp).permute(0,3,1,2)
        return trad_mask_res

    def normalize(self, out):
        min = torch.min(out)
        max = torch.max(out)
        print(min)
        print(max)

    def forward(self, image):
        # Get Matched Filtering result
        MF_result = self.filters(image)
        trad_mask, _ = torch.max(MF_result, dim=1)
        trad_mask = trad_mask.unsqueeze(1)
        trad_mask_res = self.binarize(trad_mask, 0.2)


        # For extracing image info using U-Net shaped model
        img_e1 = self.img_enc1(image)
        img_e2 = self.img_enc2(img_e1)
        img_e3 = self.img_enc3(img_e2)
        img_e4 = self.img_enc4(img_e3)
        f = self.img_enc5(img_e4)

        img_d4 = self.img_dec4(f, img_e4) # 256,1/16,1/16
        img_d3 = self.img_dec3(img_d4, img_e3) # 256,1/8,1/8
        img_d2 = self.img_dec2(img_d3, img_e2) # 128,1/4,1/4
        img_d1 = self.img_dec1(img_d2, img_e1) # 64,1/2,1/2
        img_d0 = self.img_dec0(img_d1) # 64,1,1

        # Process traditional mask obtained with matched filtering
        mask_e1 = self.mask_enc1(trad_mask) # 64,1/2,1/2
        mask_e2 = self.mask_enc2(mask_e1) # 64,1/2,1/2
        mask_e3 = self.mask_enc3(mask_e2) # 128,1/4,1/4
        mask_e4 = self.mask_enc4(mask_e3) # 256,1/8,1/8
        mask_e5 = self.mask_enc5(mask_e4) # 512,1/16,1/16

        mask_d5 = self.mask_dec5(mask_e5, img_d4) # 256, 1/16, 1/16
        mask_d5 = self.up(mask_d5)
        mask_d4 = self.mask_dec4(mask_d5, img_d3) # 128, 1/8, 1/8
        mask_d4 = self.up(mask_d4)
        mask_d3 = self.mask_dec3(mask_d4, img_d2)# 64, 1/4, 1/4
        mask_d3 = self.up(mask_d3)
        mask_d2 = self.mask_dec2(mask_d3, img_d1)# 32, 1/2, 1/2
        mask_d2 = self.up(mask_d2)
        mask_d1 = self.mask_dec1(mask_d2, img_d0)# 16, 1, 1

        # out = self.conv_img(F.leaky_relu(mask_d1, 2e-1))
        # return F.softmax(out, dim=1), trad_mask

        res = self.conv_img(F.leaky_relu(mask_d1, 2e-1))
        res = F.softmax(res, dim=1)
        # res?????????3???????????????0???????????????????????????????????????????????????1?????????????????????????????????????????????????????????????????????2??????????????????????????????????????????????????????
        # res has 3 channels, the 0th means the traditional method is correct
        # the 1st means the traditional method makes a FP mistake (mistaking the pixel non-vessel as vessel)
        # the 2nd means the traditional method makse a FN mistake (mistaking the pixel vessel as non-vessel)
        out = trad_mask_res
        out[:,0,:,:] = out[:,0,:,:] + res[:,1,:,:] - res[:,2,:,:]
        out[:,1,:,:] = out[:,1,:,:] - res[:,1,:,:] + res[:,2,:,:]
        # print(sum(sum(sum(res[:,0,:,:]))))
        # print(sum(sum(sum(res[:,1,:,:]))))
        # print(sum(sum(sum(res[:,2,:,:]))))

        out = F.softmax(out, dim=1)
        # out[out<0] = 0
        # out[out>1] = 1
        return out


from lib.pre_processing import my_PreProc

def showImg(img):
    image = Image.fromarray(img)
    image.show()


def test():
    model = MF_UNet(n_class=2, sigma=1, YLength=20)
    # model.to('cuda')
    img = cv2.imread(r'G:\Datasets\DRIVE\training\images/34_training.tif',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,[512, 512])
    img = np.reshape(img,(1,1,512,512))
    img = my_PreProc(img)

    img = torch.from_numpy(img).float()
    # img = img/255.0

    # img = img.to('cuda')
    start = time.time()
    out, trad_mask = model(img)
    end = time.time()
    #
    # out, _ = torch.max(out,dim=1)
    trad_mask = (trad_mask*255).detach().cpu().numpy()
    trad_mask = np.uint8(trad_mask)[0,0,:,:]
    _, trad_mask = cv2.threshold(trad_mask, 30, 255, cv2.THRESH_OTSU)

    showImg(trad_mask)


if __name__ == '__main__':
    test()
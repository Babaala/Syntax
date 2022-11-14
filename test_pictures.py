import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from lib.visualize import save_img, see_concat_result
import os
from lib.logger import Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
import models
from lib.common import setpu_seed
from config import parse_args


setpu_seed(2021)

class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)
        pictures_to_be_test = [(args.dir_to_be_test + '/'+ file) for file in os.listdir(args.dir_to_be_test) if file.split(".")[-1] in ['png', 'PNG', 'jpg', 'JPG']]
        with open(os.path.join(args.dir_to_be_test, "test.txt"), 'w') as f:
            for t in pictures_to_be_test:
                f.write(t)
                f.write('\n')
        args.test_data_path_list = args.dir_to_be_test + "/test.txt"

        # 设置test的
        self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = see_get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,           # 数据txt文件，里面是各个图片的位置
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

        self.lc = args.largest_connected
        self.binary_images = None

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]


    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list = see_load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            total_img = see_concat_result(self.test_imgs[i],self.pred_imgs[i], self.lc)
            # print(sum(self.pred_imgs[i]))
            save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    # 基础设置
    args = parse_args()
    # --dir_to_be_test :等待检测的图片等文件夹格式
    # --outf：输出的一级目录
    # --save：输出到二级目录
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))

    
    # 模型设置
    # net = models.UNetFamily.U_Net(1,2).to(device)
    # net = models.MF_UNet(2, 1, 10).to(device)
    # net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)
    net = models.Dense_Unet(in_chan=1).to(device)
    # net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    

    # Load checkpoint
    checkpoint = torch.load(join(save_path, 'best_model_diceloss.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)


    eval.inference(net)
    eval.evaluate()
    # print(eval.val())
    eval.save_segmentation_result()

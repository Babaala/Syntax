import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys,time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test import Test

from function import get_dataloader, train, val, get_dataloaderV2


def main():
    setpu_seed(2021)                            # 确定随机数种子
    args = parse_args()                         # 命令行参数读取
    save_path = join(args.outf, args.save)      # 分别是保存路径、项目名称
    save_args(args,save_path)                   # 保存参数

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    
    log = Logger(save_path)                     # 初始化记录表
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))                  # 设置控制台输出
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    
    # ------------------------------------------------设定不同的模型-----------------------------------------------------------------
    # net = models.UNetFamily.U_Net(1,2).to(device)
    # net = models.MF_UNet(2,1,10).to(device)
    net = models.Dense_Unet(in_chan=1).to(device)
    # net = models.MF_U_Net().to(device)
    # net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,96,96)).to(device).to(device=device))           # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')                                 # Modify default initialization method
    # net.apply(weight_init)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)


    # 是否加载与训练模型
    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1


    # 损失函数
    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    # criterion = CrossEntropyLoss2d() # Initialize loss function
    criterion = FocalLoss2d() # Initialize loss function
    # criterion = FocalLoss_t(num_classes=2)
    # criterion = DiceLoss()

    
    # 学习率调整器
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    
    
    # 数据加载
    train_loader, val_loader = get_dataloaderV2(args) # create dataloader
    # train_loader, val_loader = get_dataloader(args)
    
    # 是否在测试集上进行实验
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args) 

    best = {'epoch':0,'AUC-ROC':0.5}    # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0                         # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train_log = train(train_loader,net,criterion,optimizer,device) 
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader,net,criterion,device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()


        model_name = 'focall_loss_2d'

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model_{}.pth'.format(model_name)))
        trigger += 1
        if val_log['AUC-ROC'] > best['AUC-ROC']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model_{}.pth'.format(model_name)))
            best['epoch'] = epoch
            best['AUC-ROC'] = val_log['AUC-ROC']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC-ROC']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()

import os
import math
import argparse
# import _init_paths
from mer_spot.lib.config import cfg
from mer_spot.lib.utils.utils import fix_random_seed
from mer_spot.lib.config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random

from mer_spot.lib.dataset.TALDataset import TALDataset

from mer_spot.lib.models.fusemodel import FuseModel
from mer_spot.lib.core.function import train, evaluation
from mer_spot.lib.core.post_process import final_result_process
from mer_spot.lib.core.utils_ab import weight_init
from mer_spot.lib.utils.utils import save_model, backup_codes

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #指定使用服务器上的某张显卡


def main(subject, config):
    update_config(config)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:  #路径设置
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject)
        # 'G:\\codeassist\\lssnet\\output_V28/cas(me)^2/subject_s15'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, 'code')
        # 'G:\\codeassist\\lssnet\\output_V28/cas(me)^2/subject_s15\\code'
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)  # BACKUP_LISTS: ['lib', 'experiments', 'tools']
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # benchmark 基准
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # model
    model = FuseModel(cfg)



    model.apply(weight_init)
    model.cuda()
    
    # optimizer
    # warm_up_with_cosine_lr
    # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
      #学习率优化
    warm_up_with_cosine_lr = lambda epoch: epoch / cfg.TRAIN.WARM_UP_EPOCH if epoch <= cfg.TRAIN.WARM_UP_EPOCH else 0.5 * ( math.cos((epoch - cfg.TRAIN.WARM_UP_EPOCH) /(cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
      #torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT, subject)
    # train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
    #                           shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=0, pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT, subject)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=0, pin_memory=cfg.DATASET.PIN_MEMORY)
    # creating log path
    log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, str(cfg.TRAIN.BATCH_SIZE)+'_'+cfg.TRAIN.LOG_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)

    
    for epoch in range(cfg.TRAIN.END_EPOCH):  # END_EPOCH=30


        loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab = train(cfg, train_loader, model, optimizer)
        print('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f' % (
            epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab))


        with open(log_path, 'a') as f:
            #'G:\\codeassist\\lssnet\\output_V28/cas(me)^2/subject_s15\\32_train_loss.log'
            f.write('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f\n' % (
                epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab))


        # decay lr
        scheduler.step()
        lr = scheduler.get_last_lr()################################

        cfg.TRAIN.LR = lr


            ################################

        if (epoch+1) % cfg.TEST.EVAL_INTERVAL == 0:  # EVAL_INTERVAL: 1
            # save_model(cfg, epoch=epoch, model=model, optimizer=optimizer, subject=subject)
            # train_df_ab, train_df_af = evaluation(train_loader, model, epoch, cfg)  # 自己写的

            out_df_ab, out_df_af = evaluation(val_loader, model, epoch, cfg)

            # train_df_list = [out_df_ab, out_df_af]  # 自己写的

            out_df_list = [out_df_ab, out_df_af]

            # final_result_process(train_df_list, epoch, subject, cfg, flag=0)#自己写
            final_result_process(out_df_list, epoch, subject, cfg, flag=0)#flag0改成了1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MER SPOT')
    # parser.add_argument('--cfg', type=str, help='experiment config file', default='/home/yww/mer_spot/experiments/CAS.yaml')
    # parser.add_argument('--dataset', type=str, default='cas(me)^2')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='D:/codeassist/lssnet/mer_spot/experiments/CAS.yaml')
    parser.add_argument('--dataset', type=str, default='cas(me)^2')
    args = parser.parse_args()

    dataset = args.dataset
    new_cfg = args.cfg

    if dataset == 'cas(me)^2':
        ca_subject = [15,16,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,40]
        ca_subject = ['s'+ str(i)  for i in ca_subject]
        for i in ca_subject:
            subject = 'subject_' + i 
            main(subject, new_cfg)
    # else:
    #     # sa_subject = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,30,32,33,34,35,36,37,99]
    #     sa_subject = [15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 30, 32, 33, 34, 35, 36,37, 99]
    #     sa_subject = [str(i).zfill(3)  for i in sa_subject]
    #     for i in sa_subject:
    #         subject = 'subject_' + i
    #         main(subject, new_cfg)


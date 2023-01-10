import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TALDataset(Dataset):
    def __init__(self, cfg, split, subject):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR, subject)  # ROOT_DIR: '/home/yww/mer_spot'
          # FEAT_DIR: '/home/yww/1_spot/ca_subject_train_ori/val_test'
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT  #  TRAIN_SPLIT: 'train'
        self.target_size = (cfg.DATASET.RESCALE_TEM_LENGTH, cfg.MODEL.IN_FEAT_DIM)  # RESCALE_TEM_LENGTH = 512，IN_FEAT_DIM = 512
        self.max_segment_num = cfg.DATASET.MAX_SEGMENT_NUM  # MAX_SEGMENT_NUM = 30
        self.num_classes = cfg.DATASET.NUM_CLASSES  # NUM_CLASSES = 2
        self.base_dir = os.path.join(self.root, self.split)
        self.datas = self._make_dataset()
        self.class_label = cfg.DATASET.CLASS_IDX  # CLASS_IDX: [0, 1, 2]
        self.window_size = cfg.DATASET.WINDOW_SIZE  # WINDOW_SIZE: 128
        if self.split == self.train_split:
            self.anno_df = pd.read_csv('D:/codeassist/lssnet/mer_spot/casme2_annotation.csv')  #G:\code assist\lssnet\mer_spot\casme2_annotation.csv

        self.gt_overlap_threshold = 0.9

    def __len__(self):
        return len(self.datas)

    def get_anno(self, start_frame, video_name):
        end_frame = start_frame + self.window_size

        label = list()
        box = list()
        anno_df = self.anno_df[self.anno_df.video == video_name]
        for i in range(len(anno_df)):
            act_start = anno_df.startFrame.values[i]
            act_end = anno_df.endFrame.values[i]
            assert act_end > act_start
            overlap = min(end_frame, act_end) - max(start_frame, act_start)  # 滑窗与gt的重叠部分
            overlap_ratio = overlap * 1.0 / (act_end - act_start)

            if overlap_ratio > self.gt_overlap_threshold:  # self.gt_overlap_threshold = 0.9  # 取重叠部分大于0.9IOU
                # 重叠部分
                gt_start = max(start_frame, act_start) - start_frame  #  起始帧
                gt_end = min(end_frame, act_end) - start_frame  #  结束帧

                label.append(self.class_label.index(anno_df.type_idx.values[i]))#获取视频是mae或是 mie
                box.append([gt_start, gt_end])  # frame level

        box = np.array(box).astype('float32')
        label = np.array(label)
        return label, box

    def __getitem__(self, idx):
        file_name = self.datas[idx]  # self.datas是一个列表，1664个元素，每一个元素都是npz文件
        data = np.load(os.path.join(self.base_dir, file_name))  # data得到的是经过np.load下载的npz文件
          # base_dir = os.path.join(self.root, self.split)
          # self.root = 'G:\code assist\lssnet\mer_spot\ca_subject_train\val_test\subject_s15'
          # self.split = split % (train或test)
          # root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR, subject)
          #  ROOT_DIR: '/home/yww/mer_spot'；FEAT_DIR: '/home/yww/1_spot/ca_subject_train_ori/val_test'
          #  实际上ca_subject_train 在‘G:\code assist\lssnet\mer_spot\ca_subject_train\val_test’
        feat_tem = data['feat_tem']  # 得到1024个np.array矩阵，每个array矩阵都是一维列表，都有64个值
        # feat_tem = cv2.resize(feat_tem, self.target_size, interpolation=cv2.INTER_LINEAR)
        feat_spa = data['feat_spa']  # 得到1024个np.array矩阵，每个array矩阵都是一维列表，都有64个值
        # feat_spa = cv2.resize(feat_spa, self.target_size, interpolation=cv2.INTER_LINEAR)
        begin_frame = data['begin_frame']  # 得到一个数值，与npz文件的数值一样
        # pass video_name vis list
        video_name = str(data['vid_name'])  # 得到一个字符串，与npz文件除数值外的名称一样，如'16_0101disgustingteeth'
        
        if self.split == self.train_split:
            action = data['action']  # 得到一个二维矩阵，如[[49. 68.]]，[[ 71. 103.]]等等
            # action_tmp = [i[:2] for i in action]
            action = np.array(action).astype('float32')
            label = data['class_label']  # 得到一个只有一个数值元素的列表，值为0，1，2中的一个
            # data for anchor-based
            # label, action = self.get_anno(begin_frame, video_name)
            num_segment = action.shape[0]  # 视频的垂直尺寸  action.shape可能是(1, 2)，(2, 2)等等
            assert num_segment > 0, 'no action in {}!!!'.format(video_name)
            action_padding = np.zeros((self.max_segment_num, 2), dtype=np.float)  # 得到30*2的0值矩阵
            action_padding[:num_segment, :] = action
            label_padding = np.zeros(self.max_segment_num, dtype=np.int)  # 得到30*1的0值矩阵
            label_padding[:num_segment] = label

            return feat_spa, feat_tem, action_padding, label_padding, num_segment ,begin_frame, video_name # 我加了video_name
        else:
            return feat_spa, feat_tem, begin_frame, video_name

    def _make_dataset(self):
        datas = os.listdir(self.base_dir)  # datas列表，1664个元素，每一个元素是npz文件
        datas = [i for i in datas if i.endswith('.npz')]  # 从datas列表中取出所有的npz文件，放在datas列表中，1664个元素
        return datas


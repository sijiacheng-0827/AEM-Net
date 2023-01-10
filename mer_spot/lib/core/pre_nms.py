
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

from mer_spot.lib.core.utils_box import reg2loc

from mer_spot.lib.core.utils_af import get_targets_af
from mer_spot.lib.core.function import ab_prediction_train

def temporal_nms(train_loader, df, model, epoch, cfg,):
    '''
    temporal nms
    I should understand this process
    '''
    dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
    dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()
    for feat_spa, feat_tem, boxes, label, action_num, begin_frame, video_name in train_loader:
        feature = torch.cat((feat_spa, feat_tem), dim=1)  # 输入
        feature = feature.type_as(dtype)
        boxes = boxes.float().type_as(dtype)  # gt
        label = label.type_as(dtypel)  # gt label
        # af label
        # we do not calculate binary classification loss for anchor-free branch
        cate_label, reg_label = get_targets_af(cfg, boxes, label, action_num)
        reg_label = reg_label.type_as(dtype)  # bs, sum(t_i), 2
        cate_label = cate_label.type_as(dtype)

        out_af, out_ab = model(feature)  # anchor free 和 anchor base 的输出

        # Loss for anchor-free module, including classification loss & regression loss

        preds_cls, preds_reg = out_af

        target_loc = reg2loc(cfg, reg_label)  # 预测label

        preds_loc = reg2loc(cfg, preds_reg)  # 预测框


        anchors_x_ls, anchors_w_ls, anchors_rx_ls, anchors_rw_ls, anchors_class_ls, \
        match_xs_ls, match_ws_ls, match_scores_ls, match_labels_ls = ab_prediction_train(cfg, out_ab, label, boxes,
                                                                                         action_num)


    type_set = list(set(df.cate_idx.values[:]))  # [1, 2]
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    for t in type_set:
        if t==1:
            label = t
            tmp_df = df[df.cate_idx == t]

            start_time = np.array(tmp_df.xmin.values[:])
            end_time = np.array(tmp_df.xmax.values[:])
            scores = np.array(tmp_df.conf.values[:])

            duration = end_time - start_time
            order = scores.argsort()[::-1]
            if order[0]==match_scores_ls:


                keep = list()
                while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):  # 200
                    i = order[0]
                    keep.append(i)
                    tt1 = np.maximum(start_time[i], start_time[order[1:]])
                    tt2 = np.minimum(end_time[i], end_time[order[1:]])
                    intersection = tt2 - tt1
                    union = (duration[i] + duration[order[1:]] - intersection).astype(float)
                    iou = intersection / union

                    inds = np.where(iou <= cfg.TEST.NMS_TH)[0]  # 0.2
                    order = order[inds + 1]

                # record the result
                for idx in keep:

                    rlabel.append(label)
                    rstart.append(float(start_time[idx]))
                    rend.append(float(end_time[idx]))
                    rscore.append(scores[idx])

        else:
            label = t
            tmp_df = df[df.cate_idx == t]

            start_time = np.array(tmp_df.xmin.values[:])
            end_time = np.array(tmp_df.xmax.values[:])
            scores = np.array(tmp_df.conf.values[:])

            duration = end_time - start_time
            order = scores.argsort()[::-1]

            keep = list()
            while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):  # 200
                i = order[0]
                keep.append(i)
                tt1 = np.maximum(start_time[i], start_time[order[1:]])
                tt2 = np.minimum(end_time[i], end_time[order[1:]])
                intersection = tt2 - tt1
                union = (duration[i] + duration[order[1:]] - intersection).astype(float)
                a=0.000001
                iou = intersection / (union+a)

                inds = np.where(iou <= cfg.TEST.NMS_TH)[0]  # 0.2
                order = order[inds + 1]

            # record the result
            for idx in keep:
                if duration[idx] < 23:
                    rlabel.append(label)
                    rstart.append(float(start_time[idx]))
                    rend.append(float(end_time[idx]))
                    rscore.append(scores[idx])
    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df


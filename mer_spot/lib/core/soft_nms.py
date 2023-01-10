import numpy as np
import pandas as pd
import torch

from mer_spot.lib.core.utils_ab import tiou




def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

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

            start_time = np.array(tmp_df.xmin.values[:])  # 候选框的开始帧
            end_time = np.array(tmp_df.xmax.values[:])  # 候选框的结束帧
            scores = np.array(tmp_df.conf.values[:])  # 候选框的得分

            duration = end_time - start_time
            # order里面存放的是元素从大到小排列的索引
            order = scores.argsort()[::-1]  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)

            keep_idx = list()  # 保存最高得分的索引
            keep_sc = list()  # 保存最高得分
            # 相当于目标检测中，得到图像中每个类别目标的最高分
            score = np.array([])
            while (order.size > 0) and (len(keep_idx) < cfg.TEST.TOP_K_RPOPOSAL):
                i = order[0]  # 最大值的索引
                keep_idx.append(i)
                if len(score) > 0:
                    keep_sc.append(score[ord[0]])  # 自己加的
                else:
                    keep_sc.append(scores[i])
                if len(order) == 1:  # 自己加的
                    break
                # start_time[i]跟start_time[1:]里的每个元素逐一比较，哪个大取哪个，最终得到一个array
                tt1 = np.maximum(start_time[i], start_time[order[1:]])
                tt2 = np.minimum(end_time[i], end_time[order[1:]])
                intersection = tt2 - tt1  # 每个候选框跟索引为i的候选框重叠的帧数，得到一个array
                union = (duration[i] + duration[order[1:]] - intersection).astype(float)  # 并集
                iou = intersection / union  # 得到iou
                iou = iou.astype('float')##wojia
                order = order[1:]  # 除去最高分索引后，剩下的索引；order里都是对应scores的索引，不能用于后面的score
                ord = order
                # ord是索引正确的
                for h in range(len(order)):
                    if order[h] > i:
                        ord[h] -= 1

                if len(score) > 0:
                    score[ord] *= np.exp(-(iou * iou) / 0.8)
                    sc = score[ord]
                else:
                    scores[order] *= np.exp(-(iou * iou) / 0.8)  # 剩下候选框的分数更新
                    sc = scores[order]
                count = 0
                # 验证sc里第二个以后的分值是否比第一大，如果更大，则换到第一个位置，同时对应的索引也要换
                for j in range(len(sc)):
                    if sc[j] > sc[0]:
                        temp1 = sc[j]
                        sc[j] = sc[0]
                        sc[0] = temp1
                        score = sc
                        count += 1

                        temp2 = ord[j]
                        ord[j] = ord[0]
                        ord[0] = temp2

                        temp3 = order[j]
                        order[j] = order[0]
                        order[0] = temp3
                # 如果不存在比第一个分值大，则直接将sc赋值给score
                if count == 0:
                    score = sc

            # record the result
            keep = []
            for k in range(len(keep_sc)):
                if keep_sc[k] > 0.0004:
                    keep.append(keep_idx[k])

            for idx in keep:
                if duration[idx] > 6:
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

            keep_idx = list()
            keep_sc = list()
            score = np.array([])
            while (order.size > 0) and (len(keep_idx) < cfg.TEST.TOP_K_RPOPOSAL):

                i = order[0]
                keep_idx.append(i)
                if len(score) > 0:
                    keep_sc.append(score[ord[0]])
                else:
                    keep_sc.append(scores[i])
                if len(order) == 1:
                    break
                tt1 = np.maximum(start_time[i], start_time[order[1:]])
                tt2 = np.minimum(end_time[i], end_time[order[1:]])
                intersection = tt2 - tt1
                union = (duration[i] + duration[order[1:]] - intersection).astype(float)
                a=0.000001
                iou = intersection / (union+a)

                order = order[1:]
                ord = order
                for h in range(len(order)):
                    if order[h] > i:
                        ord[h] -= 1

                if len(score) > 0:
                    score[ord] *= np.exp(-(iou * iou) / 0.8)
                    sc = score[ord]
                else:
                    scores[order] *= np.exp(-(iou * iou) / 0.8)  # 剩下候选框的分数更新
                    sc = scores[order]
                count = 0

                for j in range(len(sc)):
                    if sc[j] > sc[0]:
                        temp1 = sc[j]
                        sc[j] = sc[0]
                        sc[0] = temp1
                        score = sc
                        count += 1

                        temp2 = ord[j]
                        ord[j] = ord[0]
                        ord[0] = temp2

                        temp3 = order[j]
                        order[j] = order[0]
                        order[0] = temp3
                if count == 0:
                    score = sc

            # record the result
            keep = []
            for k in range(len(keep_sc)):
                if keep_sc[k] > 0.001:
                    keep.append(keep_idx[k])

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


# def soft_nms(df, idx_name, cfg):
#     df = df.sort_values(by='score', ascending=False)
#     save_file = '/data/home/v-yale/ActionLocalization/output/df_sort.csv'
#     df.to_csv(save_file, index=False)
#
#     tstart = list(df.start.values[:])
#     tend = list(df.end.values[:])
#     tscore = list(df.score.values[:])
#     tcls_type = list(df.cls_type.values[:])
#     rstart = list()
#     rend = list()
#     rscore = list()
#     rlabel = list()
#
#     while len(tscore) > 0 and len(rscore) <= cfg.TEST.TOP_K_RPOPOSAL:
#         max_idx = np.argmax(tscore)
#         tmp_width = tend[max_idx] - tstart[max_idx]
#         iou = tiou(tstart[max_idx], tend[max_idx], tmp_width, np.array(tstart), np.array(tend))
#         iou_exp = np.exp(-np.square(iou) / cfg.TEST.SOFT_NMS_ALPHA)
#         for idx in range(len(tscore)):
#             if idx != max_idx:
#                 tmp_iou = iou[idx]
#                 threshold = cfg.TEST.SOFT_NMS_LOW_TH + (cfg.TEST.SOFT_NMS_HIGH_TH - cfg.TEST.SOFT_NMS_LOW_TH) * tmp_width
#                 if tmp_iou > threshold:
#                     tscore[idx] = tscore[idx] * iou_exp[idx]
#         rstart.append(tstart[max_idx])
#         rend.append(tend[max_idx])
#         rscore.append(tscore[max_idx])
#         # video class label
#         cls_type = tcls_type[max_idx]
#         label = idx_name[cls_type]
#         rlabel.append(label)
#
#         tstart.pop(max_idx)
#         tend.pop(max_idx)
#         tscore.pop(max_idx)
#         tcls_type.pop(max_idx)
#
#     new_df = pd.DataFrame()
#     new_df['start'] = rstart
#     new_df['end'] = rend
#     new_df['score'] = rscore
#     new_df['label'] = rlabel
#     return new_df


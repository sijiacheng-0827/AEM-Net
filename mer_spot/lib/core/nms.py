import numpy as np
import pandas as pd

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
        if t == 1:
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
                iou = intersection / (union+0.0001)

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
            # # print('tmp_df', tmp_df)  # 一个表
            # # print('type', type(tmp_df))  # <class 'pandas.core.frame.DataFrame'>
            # print('index', tmp_df.index)
            # # print(type(tmp_df.index))  # <class 'pandas.core.indexes.numeric.Int64Index'>
            # # print('list', list(tmp_df.index))
            # # print(type(list(tmp_df.index)))  # list
            # index1 = list(tmp_df.index)
            # # n = []
            # # for i in range(len(index1) - 1):
            # # print(tmp_df.index.duplicated())
            # i = 0
            # while i < len(index1) - 1:
            #     if index1[i + 1] - index1[i] == 2:
            #         # n.append(index1[i + 1])
            #         x = tmp_df.loc[index1[i+1], 'xmax']
            #         tmp_df.loc[index1[i], 'xmax'] = x
            #         tmp_df.drop(axis=0, index=index1[i + 1], inplace=True)
            #         index1 = list(tmp_df.index)
            #     i += 1
            #
            # # for i in n:
            # #     tmp_df.loc[i-1]['xmax'] = tmp_df.loc[i]['xmax']
            # #     tmp_df.drop(axis=0, index=i, inplace=True)



            start_time = np.array(tmp_df.xmin.values[:])
            end_time = np.array(tmp_df.xmax.values[:])
            scores = np.array(tmp_df.conf.values[:])
            duration = end_time - start_time
            order = scores.argsort()[::-1]


            # n = duration.size
            # i = 0
            # while i < n: 无效的
            #     if duration[i] > 23:
            #         # start_time[i] = start_time[i+1]
            #         start_time = np.delete(start_time, i)
            #         # end_time[i] = end_time[i + 1]
            #         end_time = np.delete(end_time, i)
            #         # scores[i] = scores[i + 1]
            #         scores = np.delete(scores, i)
            #         # duration[i] = duration[i + 1]
            #         duration = np.delete(duration, i)
            #         n -= 1
            #     i += 1







            keep = list()
            while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):  # 200

                i = order[0]
                keep.append(i)
                tt1 = np.maximum(start_time[i], start_time[order[1:]])
                tt2 = np.minimum(end_time[i], end_time[order[1:]])
                intersection = tt2 - tt1
                union = (duration[i] + duration[order[1:]] - intersection).astype(float)
                a=0.0001
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





    #     else:
    #         label = t
    #         tmp_df = df[df.cate_idx == t]
    #
    #         start_time = np.array(tmp_df.xmin.values[:])
    #         end_time = np.array(tmp_df.xmax.values[:])
    #         scores = np.array(tmp_df.conf.values[:])
    #
    #         duration = end_time - start_time
    #         order = scores.argsort()[::-1]
    #
    #         keep_idx = list()
    #         keep_sc = list()
    #         score = np.array([])
    #         while (order.size > 0) and (len(keep_idx) < cfg.TEST.TOP_K_RPOPOSAL):
    #
    #             i = order[0]
    #             keep_idx.append(i)
    #             if len(score) > 0:
    #                 keep_sc.append(score[ord[0]])
    #             else:
    #                 keep_sc.append(scores[i])
    #             if len(order) == 1:
    #                 break
    #             tt1 = np.maximum(start_time[i], start_time[order[1:]])
    #             tt2 = np.minimum(end_time[i], end_time[order[1:]])
    #             intersection = tt2 - tt1
    #             union = (duration[i] + duration[order[1:]] - intersection).astype(float)
    #             a = 0.000001
    #             iou = intersection / (union + a)
    #
    #             order = order[1:]
    #             ord = order
    #             for h in range(len(order)):
    #                 if order[h] > i:
    #                     ord[h] -= 1
    #
    #             if len(score) > 0:
    #                 score[ord] *= np.exp(-(iou * iou) / 0.38)
    #                 sc = score[ord]
    #             else:
    #                 scores[order] *= np.exp(-(iou * iou) / 0.38)  # 剩下候选框的分数更新
    #                 sc = scores[order]
    #             count = 0
    #
    #             for j in range(len(sc)):
    #                 if sc[j] > sc[0]:
    #                     temp1 = sc[j]
    #                     sc[j] = sc[0]
    #                     sc[0] = temp1
    #                     score = sc
    #                     count += 1
    #
    #                     temp2 = ord[j]
    #                     ord[j] = ord[0]
    #                     ord[0] = temp2
    #
    #                     temp3 = order[j]
    #                     order[j] = order[0]
    #                     order[0] = temp3
    #             if count == 0:
    #                 score = sc
    #
    #         # record the result
    #         keep = []
    #         for k in range(len(keep_sc)):
    #             if keep_sc[k] > 0.0001:
    #                 keep.append(keep_idx[k])
    #
    #         for idx in keep:
    #             if duration[idx] < 23:
    #                 rlabel.append(label)
    #                 rstart.append(float(start_time[idx]))
    #                 rend.append(float(end_time[idx]))
    #                 rscore.append(scores[idx])
    #
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


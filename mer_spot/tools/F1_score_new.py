import os
import glob
import pandas as pd
import numpy as np
import argparse
import math

from torch.nn.functional import threshold

def max_iou(ann_csv,part_pre,TP1,TP2,write_list):

    for video_num, pre in enumerate(part_pre):
        video_name_list = list(set(ann_csv.video.values[:].tolist()))
        video_name_list.sort()
        
        video_name_last = part_pre[video_num][0][0]
        video_name_part = 's' + video_name_last[:2]
        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)

        video_ann_df = ann_csv[ann_csv.video == video_name]
        act_start_video = video_ann_df['startFrame'].values[:]
        # act_start_video = [sorted(i, key = lambda x:int(x)) for i in act_start_video]
        
        indexes = np.argsort(act_start_video)
        act_end_video = video_ann_df['endFrame'].values[:]
        act_end_video = np.array(act_end_video)[indexes]

        labels = video_ann_df['type_idx'].values[:]
        labels = np.array(labels)[indexes]
        
        act_start_video.sort()

        # calculate f1-score
        # number of actual frames that have been calculated so far
        pre = np.array(pre)
        pre_start = pre[:,1].astype(float).astype(np.int64) * int(label_frequency)
        pre_end = pre[:,2].astype(float).astype(np.int64) * int(label_frequency)
        
        start_tmp = list()
        end_tmp = list()
        for m in range(len(act_start_video)):
            video_label = video_name_last[:7]
            act_start = int(act_start_video[m])
            act_end = int(act_end_video[m])
            iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start))/(np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start))
            max_iou = np.max(iou)
            max_index = np.argmax(iou)
            if max_iou >= 0.5 and labels[m]==int(float(pre[max_index][-2])):
                tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end, 'TP']
                write_list.append(tmp_write_list)  
                if labels[m] == 1:
                    TP1 = TP1 + 1
                elif labels[m] == 2:
                    TP2 = TP2 + 1
                start_tmp.append(pre_start[max_index])
                end_tmp.append(pre_end[max_index])
            else:
                tmp_write_list = [video_label, '_', '_', act_start, act_end, 'FP']
                write_list.append(tmp_write_list) 
        pre_start_remain = list(pre_start)
        pre_end_remain = list(pre_end)
        pre_remain_s = [i for i in pre_start_remain if i not in start_tmp] 
        pre_remain_e = [i for i in pre_end_remain if i not in end_tmp] 
        try:
            if len(pre_remain_s) == len(pre_remain_e):
                write_remain = [[video_label, i, pre_end[pre_start==i][0], '_', '_', 'FN'] for i in pre_remain_s]
                write_list = write_list + write_remain
            else:
                # print('lables in starts are repeat')
                write_remain = [[video_label, pre_start[pre_end==i][0], i, '_', '_', 'FN'] for i in pre_remain_e]
                write_list = write_list + write_remain
        except:
            pass
    return TP1,TP2,write_list

def all_score(TP1,TP2,N1,N2,recall1,recall2,recall_all):
    if TP1==0 and TP2 !=0:
        precision1 = 0
        precision2 = 1.0* TP2/N2
        precision_all = 1.0* (TP1+TP2)/(N1+N2)
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
        F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
    elif TP1!=0 and TP2 ==0: 
        precision1 = 1.0* TP1/N1 
        precision2 = 0
        precision_all = 1.0* (TP1+TP2)/(N1+N2)
        F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
        F1_SCORE_M2 = 0
        F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)
    elif TP1==0 and TP2 ==0:
        precision1 = 0
        precision2 = 0
        precision_all = 0
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 0
        F1_SCORE = 0
    else:
        precision1 = 1.0* TP1/N1
        precision2 = 1.0* TP2/N2
        precision_all = 1.0* (TP1+TP2)/(N1+N2)
        F1_SCORE_M1 = 2*(recall1*precision1)/(recall1+precision1)
        F1_SCORE_M2 = 2*(recall2*precision2)/(recall2+precision2)
        F1_SCORE = 2*(recall_all*precision_all)/(recall_all+precision_all)

    return F1_SCORE_M1, F1_SCORE_M2,F1_SCORE,precision_all

def main_topk(path, dataset, annotation, version):
    
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key = lambda x:int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()
    # txt_index = txts[-1]
    best, best_m1, best_m2 = 0, 0, 0
    if dataset =='cas(me)^2':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'catop_k'+'_'+str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'satop_k'+'_'+str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    best_out = os.path.join(out_path_tmp, 'best_sample.log')
    topk_out = os.path.join(out_path_tmp, 'topk.log')  
    if os.path.exists(topk_out):
        os.remove(topk_out)
    for e in range(4, len(txts)):
    # for e in range(25, 26):
        txt_index = txts[e]
        test_path = [os.path.join(i, 'test_'+str(txt_index).zfill(2)+'.txt') for i in test_path_temp]
        print('number of epochs:',txt_index)
        # confirm the best top_k
        for k in range(2, 15):
            standard_out = os.path.join(out_path_tmp, 'epoch'+str(e)+'_'+str(k)+'_'+'sample.log')
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            write_list = list()
            length_count = list()
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]
                
                # divide predicitons of every video
                count = 1
                tmp_list = list()
                all_test = dict()
                # no duplicate label extraction
                all_video = list(set([name[0] for name in all_lines]))
                # number of GT of every video
                num_of_video = len(all_test.keys()) 

                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # least len of GT
                # len_pre = [len(i) for i in all_test.values()]
                # least_len = min(len_pre)
                part_pre = [i[:k] for i in all_test.values()]

                # predictions: sorted by strat bondaries
                part_pre= [sorted(i, key = lambda x:int(float(x[1]))) for i in part_pre]
                
                # N1: number of precictions of macro-expressions
                # N2: number of precictions of micro-expressions
                # N_all: number of precictions
                N_all = N_all + len(part_pre) * k
                for part in part_pre:
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])

                TP1,TP2,write_list = max_iou(ann_csv,part_pre,TP1,TP2,write_list)

            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2':
                M1= 282
                M2 = 84
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0* TP1/M1
            recall2 =1.0* TP2/M2
            recall_all = 1.0 *(TP1+TP2)/(M1+M2)
            F1_SCORE_M1, F1_SCORE_M2,F1_SCORE,precision_all = all_score(TP1,TP2,N1,N2,recall1,recall2,recall_all)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                print("f1_score_macro: %05f, f1_score_micro: %05f"%(best_m1, best_m2))
            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                print("f1_score_macro: %05f, f1_score_micro: %05f"%(best_m1, best_m2))
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                print('number of epoch: %d, topk: %5f'%(e, k))
                print("recall: %05f, precision: %05f, f1_score: %05f"%(recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                if F1_SCORE > 0.25:
                    standard_out = os.path.join(out_path_tmp, str(e)+'_'+str(k)+'_'+'sample.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                    with open(topk_out, 'a') as f_threshold:
                        f_threshold.writelines("%d, %f, %d, %d, %d, %f\n" % (e, k, TP, FP, FN, F1_SCORE))
                print(TP, TP1, TP2, N1, N2)
        
      
def main_threshold(path, dataset, annotation, version, label_frequency, start_threshold,max_num_pos):
    
    files_tmp = os.listdir(path)  # files_tmp = ['subject_s15', 'subject_s16', ..., 'subject_s40']
    files = sorted(files_tmp, key = lambda x:int(x[-2:]))  # files = ['subject_s15', 'subject_s16', ..., 'subject_s40']
    ann_csv = pd.read_csv(annotation)  # ann_csv = casme2_annotation.csv
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]  # ['G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection', ..., 'G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s40/test_detection']
    test_path_temp = [i.replace('\\','/')for i in test_path_temp]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))  # ['G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection/test_00.txt', ..., 'G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection/test_29.txt']

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]  # txts = [0, 1, 2, ..., 29]
    txts.sort()  # txts = [0, 1, 2, ..., 29]

    best, best_m1, best_m2 = 0, 0, 0
    best_recall = 0
    if dataset =='cas(me)^2':
        # out_path_tmp = 'G:/codeassist/lssnet/mer_spot/threshold/cathreshold_28'
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'cathreshold'+'_'+str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'threshold', 'sathreshold'+'_'+str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    # best_out = 'G:/codeassist/lssnet/mer_spot/threshold/cathreshold_28/best_sample.log'
    best_out = os.path.join(out_path_tmp, 'best_sample.log')
    # threshold_out = 'G:/codeassist/lssnet/mer_spot/threshold/cathreshold_28/threshlod.log'
    threshold_out = os.path.join(out_path_tmp, 'threshlod.log')
    if os.path.exists(threshold_out):
        os.remove(threshold_out)  # cathreshold_28目录中如果有threshlod.log文件，则删掉
    ###############################################
    for e in range(0, 30): # orginal is （4, 60） # txts:0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        txt_index = txts[e]  # txt_index = 0, 1, ..., 29
        # all subjects in the same epoch
        # test_path = ['G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection/test_04.txt', ..., 'G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s40/test_detection/test_04.txt']
        # test_path = ['G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection/test_29.txt', ..., 'G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s40/test_detection/test_29.txt']
        test_path = [os.path.join(i, 'test_'+str(txt_index).zfill(2)+'.txt') for i in test_path_temp]
        #test_path = 'G:/codeassist/lssnet/output_V28/cas(me)^2/subject_s15/test_detection/test_00.txt'
        test_path = [i.replace('\\','/')for i in test_path]
        # confirm the best threshold
        for k_temp in range(start_threshold, 700, 1):  # start_threshold = 300
            k = 1.0 *k_temp/1000  
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            length_count = list()
            write_list = list()
            length_pre = list()
            # every subject in one file (200x)
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]  # all_lines: ['15_0402beatingpregnantwoman\t3358.656\t3374.763\t1\t0.9984\n', '15_0402beatingpregnantwoman\t4260.042\t4271.129\t1\t0.9980\n', '15_0402beatingpregnantwoman\t163.054\t172.949\t1\t0.9977\n', '15_0402beatingpregnantwoman\t983.246\t992.380\t1\t0.9977\n', '15_0402beatingpregnantwoman\t1182.196\t1189.751\t1\t0.9975\n', '15_0402beatingpregnantwoman\t113.695\t117.181\t1\t0.9975\n', '15_0402beatingpregnantwoman\t2011.142\t2016.130\t1\t0.9974\n', '15_0402beatingpregnantwoman\t3628.975\t3630.662\t1\t0.9973\n', '15_0402beatingpregnantwoman\t2662.431\t2670.048\t1\t0.9972\n', '15_0402beatingpregnantwoman\t99.042\t109.901\t1\t0.9971\n', '15_0402beatingpregnantwoman\t3049.144\t3054.797\t1\t0.9970\n', '15_0402beatingpregnantwoman\t1950.314\t1963.523\t1\t0.9968\n', '15_0402beatingpregnantwoman\t1505.324\t1527.986\t1\t0.9966\n', '15_0402beatingpregnantwoman\t3620.706\t3638.695\t1\t0.9966\n', '15_0402beatingpregnantwoman\t3118.137\t3123.625\t1\t0.9965\n', '15_0402beatingpregnantwoman\t1162.868\t1167.910\t1\t0.9964\n', '15_0402beatingpregnantwoman\t544.243\t570.320\t1\t0.9963\n', '15_0402beatingpregnantwoman\t1565.025\t1592.434\t1\t0.9961\n', '15_0402beatingpregnantwoman\t2211.075\t2224.336\t1\t0.9960\n', '15_0402beatingpregnantwoman\t811.621\t820.469\t1\t0.9959\n', '15_0402beatingpregnantwoman\t3374.901\t3378.411\t1\t0.9959\n', '15_0402beatingpregnantwoman\t4067.301\t4074.820\t1\t0.9958\n', '15_0402beatingpregnantwoman\t2725.793\t2742.847\t1\t0.9956\n', '15_0402beatingpregnantwoman\t2968.510\t2978.784\t1\t0.9955\n', '15_0402beatingpregnantwoman\t3230.748\t3238.308\t1\t0.9954\n', '15_0402beatingpregnantwoman\t1330.234\t1334.551\t1\t0.9954\n', '15_0402beatingpregnantwoman\t1257.948\t1265.713\t1\t0.9953\n', '15_0402beatingpregnantwoman\t1579.588\t1583.189\t1\t0.9953\n', '15_0402beatingpregnantwoman\t481.832\t495.702\t1\t0.9952\n', '15_0402beatingpregnantwoman\t3241.486\t3244.140\t1\t0.9952\n', '15_0402beatingpregnantwoman\t2911.358\t2931.188\t1\t0.9952\n', '15_0402beatingpregnantwoman\t2734.451\t2737.788\t1\t0.9952\n', '15_0402beatingpregnantwoman\t3172.732\t3179.235\t1\t0.9952\n', '15_0402beatingpregnantwoman\t2282.296\t2294.021\t1\t0.9951\n', '15_0402beatingpregnantwoman\t3759.196\t3762.573\t1\t0.9951\n', '15_0402beatingpregnantwoman\t3430.052\t3441.207\t1\t0.9951\n', '15_0402beatingpregnantwoman\t3091.426\t3101.322\t1\t0.9951\n', '15_0402beatingpregnantwoman\t291.993\t310.893\t1\t0.9950\n', '15_0402beatingpregnantwoman\t1448.393\t1459.714\t1\t0.9950\n', '15_0402beatingpregnantwoman\t1761.338\t1775.036\t1\t0.9950\n', '15_0402beatingpregnantwoman\t1385.805\t1395.699\t1\t0.9949\n', '15_0402beatingpregnantwoman\t3181.389\t3182.726\t1\t0.9948\n', '15_0402beatingpregnantwoman\t1833.415\t1836.879\t1\t0.9947\n', '15_0402beatingpregnantwoman\t2019.038\t2028.869\t1\t0.9946\n', '15_0402beatingpregnantwoman\t356.079\t370.261\t1\t0.9944\n', '15_0402beatingpregnantwoman\t1903.980\t1907.809\t1\t0.9943\n', '15_0402beatingpregnantwoman\t1429.049\t1444.697\t1\t0.9943\n', '15_0402beatingpregnantwoman\t1381.901\t1388.078\t1\t0.9943\n', '15_0402beatingpregnantwoman\t3027.770\t3059.038\t1\t0.9943\n', '15_0402beatingpregnantwoman\t2851.750\t2869.040\t1\t0.9943\n', '15_0402beatingpregnantwoman\t3599.332\t3608.453\t1\t0.9942\n', '15_0402beatingpregnantwoman\t1055.886\t1087.166\t1\t0.9942\n', '15_0402beatingpregnantwoman\t3955.169\t3957.353\t1\t0.9942\n', '15_0402beatingpregnantwoman\t151.172\t161.075\t1\t0.9942\n', '15_0402beatingpregnantwoman\t2846.060\t2855.743\t1\t0.9941\n', '15_0402beatingpregnantwoman\t866.981\t887.702\t1\t0.9941\n', '15_0402beatingpregnantwoman\t1310.552\t1323.700\t1\t0.9940\n', '15_0402beatingpregnantwoman\t4319.239\t4333.206\t1\t0.9939\n', '15_0402beatingpregnantwoman\t553.709\t557.463\t1\t0.9938\n', '15_0402beatingpregnantwoman\t676.515\t690.673\t1\t0.9938\n', '15_0402beatingpregnantwoman\t1116.212\t1152.000\t1\t0.9938\n', '15_0402beatingpregnantwoman\t2714.105\t2723.967\t1\t0.9938\n', '15_0402beatingpregnantwoman\t1828.529\t1846.009\t1\t0.9938\n', '15_0402beatingpregnantwoman\t2272.792\t2283.239\t1\t0.9937\n', '15_0402beatingpregnantwoman\t1685.721\t1693.862\t1\t0.9937\n', '15_0402beatingpregnantwoman\t4195.623\t4212.913\t1\t0.9935\n', '15_0402beatingpregnantwoman\t2223.282\t2225.002\t1\t0.9935\n', '15_0402beatingpregnantwoman\t3127.966\t3133.769\t1\t0.9934\n', '15_0402beatingpregnantwoman\t2986.388\t2992.239\t1\t0.9934\n', '15_0402beatingpregnantwoman\t3486.393\t3489.887\t1\t0.9934\n', '15_0402beatingpregnantwoman\t4006.593\t4014.187\t1\t0.9933\n', '15_0402beatingpregnantwoman\t3109.338\t3118.090\t1\t0.9932\n', '15_0402beatingpregnantwoman\t993.805\t1003.545\t1\t0.9931\n', '15_0402beatingpregnantwoman\t2006.902\t2012.589\t1\t0.9931\n', '15_0402beatingpregnantwoman\t4143.648\t4146.591\t1\t0.9931\n', '15_0402beatingpregnantwoman\t4056.932\t4065.304\t1\t0.9930\n', '15_0402beatingpregnantwoman\t2281.736\t2283.422\t1\t0.9930\n', '15_0402beatingpregnantwoman\t2453.566\t2471.493\t1\t0.9928\n', '15_0402beatingpregnantwoman\t95.500\t101.899\t1\t0.9928\n', '15_0402beatingpregnantwoman\t420.319\t440.908\t1\t0.9926\n', '15_0402beatingpregnantwoman\t668.913\t676.274\t1\t0.9925\n', '15_0402beatingpregnantwoman\t1175.245\t1184.534\t1\t0.9924\n', '15_0402beatingpregnantwoman\t3042.239\t3047.580\t1\t0.9924\n', '15_0402beatingpregnantwoman\t2531.420\t2544.012\t1\t0.9923\n', '15_0402beatingpregnantwoman\t1966.036\t1970.940\t1\t0.9922\n', '15_0402beatingpregnantwoman\t853.664\t859.553\t1\t0.9922\n', '15_0402beatingpregnantwoman\t2778.021\t2789.546\t1\t0.9922\n', '15_0402beatingpregnantwoman\t1700.743\t1714.007\t1\t0.9921\n', '15_0402beatingpregnantwoman\t3681.334\t3693.010\t1\t0.9920\n', '15_0402beatingpregnantwoman\t385.856\t398.457\t1\t0.9920\n', '15_0402beatingpregnantwoman\t2335.528\t2357.485\t1\t0.9920\n', '15_0402beatingpregnantwoman\t3822.313\t3825.607\t1\t0.9919\n', '15_0402beatingpregnantwoman\t1880.454\t1906.110\t1\t0.9918\n', '15_0402beatingpregnantwoman\t2026.870\t2034.632\t1\t0.9917\n', '15_0402beatingpregnantwoman\t1713.633\t1717.876\t1\t0.9916\n', '15_0402beatingpregnantwoman\t4200.702\t4202.066\t1\t0.9916\n', '15_0402beatingpregnantwoman\t4230.539\t4246.100\t1\t0.9914\n', '15_0402beatingpregnantwoman\t2520.844\t2531.832\t1\t0.9914\n', '15_0402beatingpregnantwoman\t2908.455\t2914.981\t1\t0.9913\n', '15_0402beatingpregnantwoman\t1263.963\t1267.657\t1\t0.9913\n'...
                
                # divide all gts of every video
                # tmp_video = all_lines[0][0]
                count = 1
                tmp_list = list()
                all_test = dict()
                all_video = list(set([name[0] for name in all_lines]))  # <class 'list'>: ['15_0101disgustingteeth', '15_0402beatingpregnantwoman', '15_0505funnyinnovations', '15_0503unnyfarting', '15_0401girlcrashing', '15_0102eatingworms', '15_0502funnyerrors']
                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):  # j=200 跳转
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # number of GT of every video
                num_of_video = len(all_test.keys())   # num of video = 7
                
                # least len of GT
                part_tmp = list()
                # select predictions of every video (prob > threshold)
                for i in range(num_of_video):  # 0-7
                    tmp_one_video = list(all_test.values())[i]
                    for o in tmp_one_video:
                        if o[-1][:-2]=='':
                            print(o)
                    part = [o for o in tmp_one_video if float(o[-1][:-2]) > k]

                    # N1: number of precictions of macro-expressions
                    # N2: number of precictions of micro-expressions
                    # N_all: number of precictions
                    if len(part) > max_num_pos :
                        part = part[:max_num_pos]  # 取前15个
                    N_all = N_all + len(part)  # 0+15
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])

                    if not part:
                        part = [[tmp_one_video[0][0], '100000', '100000', '_','_']]
                    part_tmp.append(part)
                part_pre = part_tmp

                # predictions: sorted by prob
                part_pre= [sorted(i, key = lambda x:int(float(x[1]))) for i in part_pre]
                
                # calculate iou between every prediction with GT
                for video_num, pre in enumerate(part_pre):
                    video_name_list = list(set(ann_csv.video.values[:].tolist()))
                    video_name_list.sort()
                    
                    # identify the current video
                    video_name_last = part_pre[video_num][0][0]  # '15_0101disgustingteeth'
                    if dataset =='cas(me)^2':
                        video_name_part = 's' + video_name_last[:2]  # 's15'
                        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)

                    else:
                        video_name = os.path.join(video_name_list[0][:-4],str(video_name_last).zfill(3))

                    # select startframes of current video

                    video_name = video_name.replace('\\', '/')
                    video_ann_df = ann_csv[ann_csv.video == video_name]
                    # 出现错误 './CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped\\s15\\15_0505funnyinnovations'
                    act_start_video = video_ann_df['startFrame'].values[:]  # 0：138 1：178 2：2203
                    # select indexes of startframes of current video
                    indexes = np.argsort(act_start_video)
                    # labels and endframes are sorted by indexes from actual start frames
                    act_end_video = video_ann_df['endFrame'].values[:]  # 0：148 1：190 2：2227
                    act_end_video = np.array(act_end_video)[indexes]
                    labels = video_ann_df['type_idx'].values[:]
                    labels = np.array(labels)[indexes]
                    # actual start frames are sorted by time series
                    act_start_video.sort()
                    
                    pre = np.array(pre)
                    pre_start = pre[:,1].astype(float).astype(np.int64) * int(label_frequency)
                    pre_end = pre[:,2].astype(float).astype(np.int64) * int(label_frequency)

                    start_tmp = list()
                    end_tmp = list()
                    for m in range(len(act_start_video)):  #  出现错误
                        video_label = video_name_last[:7]
                        act_start = int(act_start_video[m])
                        act_end = int(act_end_video[m])
                        iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start))/(np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start))
                        max_iou = np.max(iou)
                        max_index = np.argmax(iou)
                        if max_iou >= 0.5 and labels[m]==int(float(pre[max_index][-2])):
                            tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end, 'TP']
                            write_list.append(tmp_write_list)  
                            if labels[m] == 1:
                                TP1 = TP1 + 1
                            elif labels[m] == 2:
                                TP2 = TP2 + 1
                            start_tmp.append(pre_start[max_index])
                            end_tmp.append(pre_end[max_index])
                        else:
                            tmp_write_list = [video_label, '_', '_', act_start, act_end, 'FP']
                            write_list.append(tmp_write_list) 
                    pre_start_remain = list(pre_start)
                    pre_end_remain = list(pre_end)
                    pre_remain_s = [i for i in pre_start_remain if i not in start_tmp] 
                    pre_remain_e = [i for i in pre_end_remain if i not in end_tmp] 
                    try:
                        if pre_remain_s[0] < 100000 and len(pre_remain_s) == len(pre_remain_e):
                            write_remain = [[video_label, i, pre_end[pre_start==i][0], '_', '_', 'FN'] for i in pre_remain_s]
                            write_list = write_list + write_remain
                        elif pre_remain_s[0] == 100000:
                            pass
                        else:
                            # print('lables in starts are repeat')
                            write_remain = [[video_label, pre_start[pre_end==i][0], i, '_', '_', 'FN'] for i in pre_remain_e]
                            write_list = write_list + write_remain
                    except:
                        pass
            
            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2' or dataset == 'cas(me)^2_merge':
                M1= 282
                M2 = 84
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0* TP1/M1
            recall2 =1.0* TP2/M2
            recall_all = 1.0 *(TP1+TP2)/(M1+M2)
            if recall_all > best_recall:
                best_recall = recall_all
                print('best', recall_all)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            F1_SCORE_M1, F1_SCORE_M2,F1_SCORE,precision_all = all_score(TP1,TP2,N1,N2,recall1,recall2,recall_all)
            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                print("f1_score_macro: %05f, f1_score_micro: %05f"%(best_m1, best_m2), "\n")
            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                print("f1_score_macro: %05f, f1_score_micro: %05f"%(best_m1, best_m2), "\n")
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                # print('number of epoch: %d, threshold: %5f'%(e, k))
                print("recall: %05f, precision: %05f, f1_score: %05f"%(recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                if F1_SCORE > 0.25:
                    standard_out = os.path.join(out_path_tmp, str(e)+'_'+str(k)+'_'+'sample.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines("%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1],wtmp[2],wtmp[3],wtmp[4],wtmp[5]) for wtmp in write_list)
                    with open(threshold_out, 'a') as f_threshold:
                        f_threshold.writelines("%d, %f, %d, %d, %d, %f\n" % (e, k, TP, FP, FN, F1_SCORE))
                length_count.sort()
                length_pre.sort()
                # print('pre:', length_pre,'\n','act:', length_count,'\n',TP, TP1, TP2, N1, N2)
                print(TP, TP1, TP2, N1, N2)
            # print(TP, TP1, TP2, N1, N2,k_temp/1000)
        print("epoch:  !!!!!!!!!!!!!!!!!!!!!!!!", e+1)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--path', type=str, default='D:/codeassist/lssnet/output_V28/cas(me)^2')
    parser.add_argument('--ann', type=str, default='D:/codeassist/lssnet/mer_spot/casme2_annotation.csv')
    parser.add_argument('--dataset', type=str, default='cas(me)^2')
    # # parser.add_argument('--ann', type=str, default=r'/home/yww/1_spot/cas(me)^2_merge.csv')
    # # parser.add_argument('--dataset', type=str, default=r'cas(me)^2_merge')
    parser.add_argument('--version', type=int, default=28)
    parser.add_argument('--top_k', type=bool, default=False)
    parser.add_argument('--label_frequency', type=float, default=1.0)
    parser.add_argument('--start_threshold', type=int, default=300)
    parser.add_argument('--most_pos_num', type=int, default=15)

    # parser.add_argument('--path', type=str, default='/home/yww/1_spot/MSA-Net/output_V28/samm_5')
    # parser.add_argument('--ann', type=str, default='/home/yww/1_spot/samm_annotation_merge.csv')
    # parser.add_argument('--dataset', type=str, default='samm')
    # parser.add_argument('--version', type=int, default=28)
    # parser.add_argument('--top_k', type=bool, default=False)
    # parser.add_argument('--label_frequency', type=float, default=5.0)
    # parser.add_argument('--start_threshold', type=int, default=100)
    # parser.add_argument('--most_pos_num', type=int, default=70)
    
    args = parser.parse_args()

    path = args.path  # G:/codeassist/lssnet/output_V28/cas(me)^2
    dataset = args.dataset  # cas(me)^2
    ann = args.ann  # G:/codeassist/lssnet/mer_spot/casme2_annotation.csv
    version = args.version  # 28
    top_k = args.top_k  # False
    label_frequency = args.label_frequency  # 1.0
    start_threshold = args.start_threshold  # 300
    max_num_pos = args.most_pos_num  # 15
    if top_k:
        main_topk(path, dataset, ann, version)
    else:
        main_threshold(path, dataset, ann, version, label_frequency, start_threshold, max_num_pos)
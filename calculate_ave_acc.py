import numpy as np
import pickle


def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    # print(idxs)
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    # print(Yi_split)
    return Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    # print(intervals)
    return intervals


def overlap_f( p, y, n_classes, overlap):
    true_intervals = np.array(segment_intervals(y))
    true_labels = segment_labels(y).astype(int)
    pred_intervals = np.array(segment_intervals(p))
    pred_labels = segment_labels(p).astype(int)
    # print('true',true_intervals, true_labels)
    # print('pre',pred_intervals, pred_labels)


    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]
    # print(n_true, n_pred)


    TP = np.zeros(n_classes, np.float)
    FP = np.zeros(n_classes, np.float)
    true_used = np.zeros(n_true, np.float)

    for j in range(n_pred):
        # Compute IoU against all others
        intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0],
                                                                                           true_intervals[:, 0])
        union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0],
                                                                                    true_intervals[:, 0])
        IoU = (intersection / union) * (pred_labels[j] == true_labels)
        # print(IoU)

        # Get the best scoring segment
        idx = IoU.argmax()

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        if IoU[idx] >= overlap and not true_used[idx]:
            TP[pred_labels[j]] += 1
            true_used[idx] = 1
        else:
            FP[pred_labels[j]] += 1
    # print(TP,FP,true_used,6666666)
    TP = TP.sum()
    FP = FP.sum()
    # False negatives are any unused true segment (i.e. "miss")
    FN = n_true - true_used.sum()

    return TP, FP, FN


def test_Ave_Seg(filename,interval_len,f1_param):
    with open(filename+'.kpl', 'rb') as handle:
        data_temp = pickle.load(handle)
    print('data len is: ', len(data_temp))

    # interval_len = 20
    # f1_param = 0.5

    activity_num = 11
    correct_count = 0
    test_sum = 0
    all_TP, all_FP, all_FN = 0, 0, 0
    for one_data in data_temp:
        csi_index = one_data[0]
        csi_flag = one_data[1]
        activity_pre = one_data[2]

        # print(csi_index,csi_flag,'truth')
        # print('pre list',activity_pre)

        pre_index = []
        pre_flag = []
        for i in range(1, len(activity_pre)):
            if activity_pre[i - 1] != activity_pre[i]:
                pre_index.append(i * interval_len)
                pre_flag.append(activity_pre[i - 1])
        pre_index.append(len(activity_pre) * interval_len)
        pre_flag.append(activity_pre[-1])



        creat_pre_flag = np.zeros((pre_index[-1],))
        start_index = 0
        for i in range(len(pre_flag)):
            creat_pre_flag[start_index:pre_index[i]] = pre_flag[i]
            # print(pre_flag[i])
            start_index = pre_index[i]
        # print(creat_pre_flag.shape,creat_pre_flag,'creat_pre_flag')

        creat_csi_flag = np.zeros((csi_index[-1] - csi_index[0],))
        start_index = 0
        for i in range(len(csi_flag)):
            creat_csi_flag[start_index:(start_index + csi_index[i + 1] - csi_index[i])] = csi_flag[i]
            start_index = start_index + csi_index[i + 1] - csi_index[i]
        # print(creat_csi_flag.shape,'creat_csi_flag')

        leaf_len = creat_csi_flag.shape[0] - creat_pre_flag.shape[0]
        # print(leaf_len,'leaf_len')
        add_csi_flag = np.ones((creat_csi_flag.shape[0] - creat_pre_flag.shape[0],)) * 20  # 20 is a abnormal value
        creat_pre_flag = np.concatenate((creat_pre_flag, add_csi_flag), axis=0)
        # print(len(creat_csi_flag),len(creat_pre_flag),1111111)

        consult = (creat_pre_flag == creat_csi_flag)
        # print(consult)
        correct_count += np.sum(consult)
        test_sum += consult.shape[0]
        if leaf_len != 0:
            yi = creat_csi_flag[0:-leaf_len]
            pi = creat_pre_flag[0:-leaf_len]
        else:
            yi = creat_csi_flag
            pi = creat_pre_flag
        # print(yi.shape, pi.shape,111111111)
        TP, FP, FN = overlap_f(pi, yi, activity_num, f1_param)
        all_TP += TP
        all_FP += FP
        all_FN += FN
    #
    return all_TP,all_FP,all_FN,correct_count,test_sum


if __name__ == '__main__':

    filename = ['TCN_based', 'LSTM_based','NO_OSSM']
    for m in range(0,2):
        if m==0:
            f1_param = 0.50
        else:
            f1_param = 0.75

        for n in range(0,3):
            print('--------------' + filename[n] + '----------------')
            all_TP, all_FP, all_FN, all_correct_count, sum = 0, 0, 0, 0, 0
            for i in range(1, 3, 1):
                TP, FP, FN, correct_count, test_sum = test_Ave_Seg(
                    'ave_accuracy/envir' + str(i) + '/' + filename[n] + '/person in trainset test', 20, f1_param)
                all_TP += TP
                all_FP += FP
                all_FN += FN
                all_correct_count += correct_count
                sum += test_sum
            print('basic '+filename[n]+' test:')
            print('average package-wise accuracy is :', all_correct_count / sum)
            # print('all_TP,all_FP,all_FN ',all_TP,all_FP,all_FN)
            precision = all_TP / (all_TP + all_FP)
            recall = all_TP / (all_TP + all_FN)
            F1 = 2 * (precision * recall) / (precision + recall)
            # If the prec+recall=0, it is a NaN. Set these to 0.
            F1 = np.nan_to_num(F1)
            print('average F1@' + str(f1_param) + ' is ', F1)

            print('--------------')
            all_TP, all_FP, all_FN, all_correct_count, sum = 0, 0, 0, 0, 0
            for i in range(1, 3, 1):
                TP, FP, FN, correct_count, test_sum = test_Ave_Seg(
                    'ave_accuracy/envir' + str(i) + '/' + filename[n] + '/person not in trainset test', 20, f1_param)
                all_TP += TP
                all_FP += FP
                all_FN += FN
                all_correct_count += correct_count
                sum += test_sum
            print('cross_subject '+filename[n]+' test:')
            print('average package-wise accuracy is :', all_correct_count / sum)
            # print('all_TP,all_FP,all_FN ',all_TP,all_FP,all_FN)
            precision = all_TP / (all_TP + all_FP)
            recall = all_TP / (all_TP + all_FN)
            F1 = 2 * (precision * recall) / (precision + recall)
            F1 = np.nan_to_num(F1)
            print('average F1@' + str(f1_param) + ' is ', F1)

            print('--------------'+filename[n]+'----------------')
            print()














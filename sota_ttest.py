import sys, os, json
import numpy as np

from scipy import stats

if __name__ == "__main__":
    loggger_path = sys.argv[1]
    dataset = sys.argv[2]

    if dataset == 'AESRC':
        with open(loggger_path) as json_data:
            cur_result_list = json.load(json_data)

        uar_list = [cur_result_list[i]['UAR'] for i in range(len(cur_result_list))]
        acc_list = [cur_result_list[i]['Accuracy'] / 100.0 for i in range(len(cur_result_list))]

        uar_interval = stats.t.interval(0.95, len(uar_list)-1, loc=np.mean(uar_list), scale=stats.sem(uar_list))
        acc_interval = stats.t.interval(0.95, len(acc_list)-1, loc=np.mean(acc_list), scale=stats.sem(acc_list))

        print ("uar: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(uar_interval[0], uar_interval[1]
            , np.mean(uar_list), np.mean(uar_list) - uar_interval[0]))
        print ("acc: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(acc_interval[0], acc_interval[1]
            , np.mean(acc_list), np.mean(acc_list) - acc_interval[0]))

        result = stats.ttest_1samp(acc_list, popmean=0.8363, alternative='greater')
        print ("One-sample t-test for Accuracy:", result)

    elif dataset == 'VCTK':
        with open(loggger_path) as json_data:
            cur_result_list = json.load(json_data)

        uar_list = [cur_result_list[i]['UAR'] for i in range(len(cur_result_list))]
        acc_list = [cur_result_list[i]['Accuracy'] / 100.0 for i in range(len(cur_result_list))]

        uar_interval = stats.t.interval(0.95, len(uar_list)-1, loc=np.mean(uar_list), scale=stats.sem(uar_list))
        acc_interval = stats.t.interval(0.95, len(acc_list)-1, loc=np.mean(acc_list), scale=stats.sem(acc_list))

        print ("uar: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(uar_interval[0], uar_interval[1]
            , np.mean(uar_list), np.mean(uar_list) - uar_interval[0]))
        print ("acc: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(acc_interval[0], acc_interval[1]
            , np.mean(acc_list), np.mean(acc_list) - acc_interval[0]))

        result = stats.ttest_1samp(uar_list, popmean=0.356, alternative='greater')
        print ("One-sample t-test for UAR:", result)
        
    else:
        print ('Invalid dataset name.')
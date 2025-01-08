import sys, os, json
import numpy as np

from scipy import stats

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loggger_path = sys.argv[1]
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

        print ('{:.2f}\\textpm{:.2f} & {:.2f}\\textpm{:.2f}'.format(np.mean(uar_list) * 100.0
            , (np.mean(uar_list) - uar_interval[0]) * 100.0, np.mean(acc_list) * 100.0, (np.mean(acc_list) - acc_interval[0]) * 100.0))
    else:
        loggger_path1 = sys.argv[1]
        with open(loggger_path1) as json_data:
            cur_result_list1 = json.load(json_data)

        uar_list1 = [cur_result_list1[i]['UAR'] for i in range(len(cur_result_list1))]
        acc_list1 = [cur_result_list1[i]['Accuracy'] / 100.0 for i in range(len(cur_result_list1))]

        loggger_path2 = sys.argv[2]
        with open(loggger_path2) as json_data:
            cur_result_list2 = json.load(json_data)

        uar_list2 = [cur_result_list2[i]['UAR'] for i in range(len(cur_result_list2))]
        acc_list2 = [cur_result_list2[i]['Accuracy'] / 100.0 for i in range(len(cur_result_list2))]

        result = stats.ttest_ind(uar_list1, uar_list2, equal_var=False, nan_policy='propagate', alternative='greater')
        print ("Ind t-test for UAR: #1 > #2 result:", result)

        result = stats.ttest_ind(acc_list1, acc_list2, equal_var=False, nan_policy='propagate', alternative='greater')
        print ("Ind t-test for Accuracy: #1 > #2 result:", result)
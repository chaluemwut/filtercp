from result_printing import print_all_result
import pickle
import matplotlib.pyplot as plt
import numpy as np

# def perf_boxplot():
#     all_result = pickle.load(open('data/result/result400_v2.data', 'rb'))
#     plt.boxplot([all_result['perf_text'], all_result['perf_ml'], all_result['perf_ml_word']])
#     fig = plt.figure(1, figsize=(9, 6))
#     ax = fig.add_subplot(111)
#     ax.set_xticklabels(['Topic Feature', 'Text Features', 'Social Media Features'])
#     plt.show()


def new_main_plot():
    all_result = pickle.load(open('data/newresult/result/all_result.obj', 'rb'))
    plt.boxplot([all_result['f1_topic_lst'], all_result['f1_text_lst'], all_result['f1_social_lst']])
    fig = plt.figure(1, figsize=(2, 2))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['Topic Feature', 'Text Features', 'Social Media Features'])
    plt.ylabel('F1-Score value')
    plt.show()

if __name__ == '__main__':
    new_main_plot()

    # all_result = pickle.load(open('data/result/resize.data', 'rb'))
    # str_ml = "time_train_ml"
    # str_ml_word = "time_predict_ml_word"
    # str_topic = "time_predict_topic"
    # str_text = "time_predict_text"
    # result25 = all_result['0.25']
    # result50 = all_result['0.5']
    # result75 = all_result['0.75']
    # print('{},{},{},{}'.format(np.average(result75[str_ml]),\
    #                      np.average(result75[str_ml_word]),\
    #                      np.average(result75[str_topic]),\
    #                      np.average(result75[str_text])
    #                      ))
    # print('{},{},{},{}'.format(np.average(result50[str_ml]),\
    #                      np.average(result50[str_ml_word]),\
    #                      np.average(result50[str_topic]),\
    #                      np.average(result50[str_text])
    #                      ))
    # print('{},{},{},{}'.format(np.average(result25[str_ml]),\
    #                      np.average(result25[str_ml_word]),\
    #                      np.average(result25[str_topic]),\
    #                      np.average(result25[str_text])
    #                      ))
    # print(result25)
    # result50 = all_result[.50]
    # result75 = all_result[.75]
    # print(result25)
    # print_all_result(all_result)
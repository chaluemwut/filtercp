import pickle
from scipy import stats

def process():
    cos_result = pickle.load(open('data/all_result/all_result_cos.obj', 'rb'))
    ml_result = pickle.load(open('data/all_result/all_result_ml.obj', 'rb'))
    data_list = [
                # cos_result['f1_topic_lst'], \
                # cos_result['f1_text_lst'], \
                #
                # cos_result['f1_topic_and_text_lst'], \
                # cos_result['f1_social_lst'], \
                #
                # cos_result['f1_social_and_text_lst'], \
                # cos_result['f1_topic_and_social_lst'], \
                # cos_result['f1_topic_text_social_lst'],\

                ml_result['f1_topic_lst'], \
                ml_result['f1_text_lst'], \

                ml_result['f1_topic_and_text_lst'], \
                ml_result['f1_social_lst'],\

                ml_result['f1_topic_and_social_lst'],\
                ml_result['f1_social_and_text_lst'],\
                ml_result['f1_topic_text_social_lst']
                ]

    # s = stats.shapiro(cos_result['f1_topic_lst'])
    # print(s)

    d1 = stats.ttest_ind(ml_result['f1_social_lst'], ml_result['f1_topic_lst'])
    print(d1)

    d1 = stats.ttest_ind(ml_result['f1_social_lst'], ml_result['f1_text_lst'])
    print(d1)

    d1 = stats.ttest_ind(ml_result['f1_social_lst'], ml_result['f1_topic_and_social_lst'])
    print(d1)

    d1 = stats.ttest_ind(ml_result['f1_social_lst'], ml_result['f1_social_and_text_lst'])
    print(d1)

    d1 = stats.ttest_ind(ml_result['f1_social_lst'], ml_result['f1_topic_text_social_lst'])
    print(d1)

    # for d in data_list:
    #     s = stats.shapiro(d)
    #     print(s)

if __name__ == '__main__':
    process()
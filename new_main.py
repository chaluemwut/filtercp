import pickle, random, time, logging, sys, json, codecs
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp import CRFWordSegment
from utilfile import FileUtil
from sklearn.metrics import f1_score
import numpy as np
from data_bean import NewDataMapping

dict_list = set([x.replace('\n', '') for x in FileUtil.read_file('data/resource/dict.txt')])

log = logging.getLogger('newmain')
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = logging.FileHandler("newmain.log")
fh.setFormatter(format)
log.addHandler(fh)

def social_and_text_feature_process(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.social_and_text_features) for x in x_train]
    x_test = [np.array(x.social_and_text_features) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

def social_feature_process(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.social_features) for x in x_train]
    x_test = [np.array(x.social_features) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

def topic_feature_process(x_train, x_test, y_train, y_test):
    x_train_msg = []
    x_test_msg = []
    crf = CRFWordSegment()
    for x_msg in x_train:
        data_lst = crf.crfpp(x_msg.message)
        data_msg = ' '.join(data_lst)
        x_train_msg.append(data_msg)

    for x_msg in x_test:
        data_lst = crf.crfpp(x_msg.message)
        data_msg = ' '.join(data_lst)
        x_test_msg.append(data_msg)

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier())])

    text_clf = text_clf.fit(x_train_msg, y_train)
    y_pred = text_clf.predict(x_test_msg)

    f1 = f1_score(y_test, y_pred)
    return f1

def text_feature_process(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.text_features) for x in x_train]
    x_test = [np.array(x.text_features) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

def load_data():
    print('start...')
    nlp = CRFWordSegment()
    with codecs.open('data/db/filterel4000.json', 'r', encoding='utf8') as f:
        lines = f.readlines()
        data_obj = []
        for data in lines:
            json_data = json.loads(data)
            if json_data['cred_value'] == 'maybe' or json_data['tag_with'] == 'NaN':
                continue

            mapping = NewDataMapping()
            message = json_data['message']
            mapping.message = message
            if json_data['cred_value'] == 'no':
                mapping.prediction_result = 0
            else:
                mapping.prediction_result = 1
            social_features = []
            social_features.append(int(json_data['likes']))
            social_features.append(int(json_data['shares']))
            social_features.append(int(json_data['comments']))
            social_features.append(int(json_data['url']))
            social_features.append(int(json_data['hashtag']))
            social_features.append(int(json_data['images']))
            social_features.append(int(json_data['vdo']))
            social_features.append(int(json_data['location']))
            social_features.append(int(json_data['non_location']))
            social_features.append(int(json_data['share_only_friend']))
            social_features.append(int(json_data['is_public']))
            social_features.append(int(json_data['feeling_status']))
            social_features.append(int(json_data['tag_with']))
            mapping.social_features = social_features

            text_features = []
            text_features.append(len(message))
            text_features.append(message.count('?'))
            text_features.append(message.count('!'))
            message_lst = nlp.crfpp(message)
            number_in_dict = dict_list & set(message_lst)
            out_side_dict = len(message_lst) - len(number_in_dict)
            text_features.append(len(message_lst))
            text_features.append(len(number_in_dict))
            text_features.append(out_side_dict)
            mapping.text_features = text_features

            social_and_text_features = []
            social_and_text_features.extend(social_features)
            social_and_text_features.extend(text_features)
            mapping.social_and_text_features = social_and_text_features

            data_obj.append(mapping)

    pickle.dump(data_obj, open('data/newresult/data/data_obj.obj', 'wb'))
    return data_obj

def process():
    mapping_lst = load_data()
    x = []
    y = []
    for mapping in mapping_lst:
        x.append(mapping)
        y.append(mapping.prediction_result)

    f1_text_lst = []
    f1_topic_lst = []
    f1_social_lst = []
    f1_social_and_text_lst = []
    for i in range(0, 100):
        log.info('****** start loop {} '.format(i))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randrange(1000))

        f1_text = text_feature_process(x_train, x_test, y_train, y_test)
        f1_topic = topic_feature_process(x_train, x_test, y_train, y_test)
        f1_social = social_feature_process(x_train, x_test, y_train, y_test)
        f1_social_and_text = social_and_text_feature_process(x_train, x_test, y_train, y_test)

        f1_text_lst.append(f1_text)
        f1_topic_lst.append(f1_topic)
        f1_social_lst.append(f1_social)
        f1_social_and_text_lst.append(f1_social_and_text)

    all_result = {}
    all_result['f1_text_lst'] = f1_text_lst
    all_result['f1_topic_lst'] = f1_topic_lst
    all_result['f1_social_lst'] = f1_social_lst
    all_result['f1_social_and_text_lst'] = f1_social_and_text_lst
    pickle.dump(all_result, open('data/newresult/result/all_result.obj', 'wb'))

if __name__ == '__main__':
    process()
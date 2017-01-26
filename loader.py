import json, pickle
from data_bean import MappingData
from nlp import CRFWordSegment
from utilfile import FileUtil

lst_param = ['cred_value', 'message', 'likes',
             'shares', 'comments', 'views', 'url',
             'hashtag', 'images', 'vdo', 'location', 'non_location',
             'is_public', 'share_only_friend', 'feeling_status',
             'tag_with', 'user_evaluator']

dict_list = set([x.replace('\n', '') for x in FileUtil.read_file('data/resource/dict.txt')])

def isInt(str_float):
    try:
        int(str_float)
        return True
    except Exception as e:
        return False

def covertToInt(str):
    return int(str)

def covertToFloat(str):
    f = float(str)
    return f*1000

def load_data():
    print('start...')
    nlp = CRFWordSegment()
    with open('data/db/filterel4000.json') as f:
        lines = f.readlines()
        data_obj = []
        for data in lines:
            json_data = json.loads(data)
            if json_data['cred_value'] == 'maybe' or json_data['tag_with'] == 'NaN':
                continue

            mapping = MappingData()
            message = json_data['message']
            mapping.message = message
            if json_data['cred_value'] == 'no':
                mapping.prediction_result = 0
            else:
                mapping.prediction_result = 1
            feature_data = []
            feature_data.append(int(json_data['likes']))
            feature_data.append(int(json_data['shares']))
            feature_data.append(int(json_data['comments']))
            feature_data.append(int(json_data['url']))
            feature_data.append(int(json_data['hashtag']))
            feature_data.append(int(json_data['images']))
            feature_data.append(int(json_data['vdo']))
            feature_data.append(int(json_data['location']))
            feature_data.append(int(json_data['non_location']))
            feature_data.append(int(json_data['share_only_friend']))
            feature_data.append(int(json_data['is_public']))
            feature_data.append(int(json_data['feeling_status']))
            feature_data.append(int(json_data['tag_with']))
            mapping.feature_list = feature_data

            feature_and_word_data = feature_data[:]
            feature_and_word_data.append(len(message))
            feature_and_word_data.append(message.count('?'))
            feature_and_word_data.append(message.count('!'))

            message_lst = nlp.crfpp(message)
            number_in_dict = dict_list & set(message_lst)
            out_side_dict = len(message_lst) - len(number_in_dict)
            feature_and_word_data.append(len(message_lst))
            feature_and_word_data.append(len(number_in_dict))
            feature_and_word_data.append(out_side_dict)
            mapping.feature_and_word_list = feature_and_word_data
            data_obj.append(mapping)

        pickle.dump(data_obj, open('data/obj/data4000.obj', 'wb'))
        print('end load...')


if __name__ == '__main__':
    # load_data()
    a = [1,2,3]
    b = [3,4,5]
    c = [2,3,4]
    for aa,bb,cc in zip(a,b,c):
        print(aa,bb,cc)
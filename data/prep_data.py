import os
import json
import csv
import numpy as np
import random
import pandas as pd

import torch
import torch.nn.functional as F


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


def change_path(json_file_path, target_path):
    with open(json_file_path, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']

    # change the path in the json file
    for i in range(len(data)):
        ori_path = data[i]["wav"]
        new_path = target_path + '/audio_16k/' + ori_path.split('/')[-1]
        data[i]["wav"] = new_path

    with open(json_file_path, 'w') as f:
        json.dump({'data': data}, f, indent=1)


def alter_data_path(data_dir):
    # for train, validation, test
    json_files = get_immediate_files(data_dir + '/datafiles/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/' + json_f)
            change_path(data_dir + '/datafiles/' + json_f, data_dir)

    # for subtest sets
    json_files = get_immediate_files(data_dir + '/datafiles/subtest/')
    for json_f in json_files:
        if json_f.endswith('.json'):
            print('now processing ' + data_dir + '/datafiles/subtest/' + json_f)
            change_path(data_dir + '/datafiles/subtest/' + json_f, data_dir)

    print('successful change the files path!')


def make_mid_dict(label_csv):
    labels_mid = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            labels_mid[row['mid']] = []
    return labels_mid


def data_set_split(all_json_file_path, target_path, class_labels_indices_path, valid_ratio,
                   split_rate=0, split_number=0):
    # read all_json file context
    with open(all_json_file_path, 'r') as all_fp:
        all_data_json = json.load(all_fp)
    all_data = all_data_json['data']
    random.shuffle(all_data)

    # dict to pandas
    # all_data_pd = pd.DataFrame(all_data)

    # read class_labels_indices_vs file
    labels_mid = make_mid_dict(class_labels_indices_path)

    for i in range(len(all_data)):
        item = all_data[i]
        labels_mid[item['labels']].append(item)

    # create number list
    number_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    data_split = {}
    for i in range(split_number):
        data_number = 'data_' + number_list[i]
        data_number_tra = 'data_' + number_list[i] + '_tra'
        data_number_val = 'data_' + number_list[i] + '_val'
        data_split[data_number] = []
        data_split[data_number_tra] = []
        data_split[data_number_val] = []

    for k in labels_mid:
        data_temp = labels_mid[k]
        len_v = len(data_temp)
        split_len = len_v // split_number
        split_step = 0
        for i in range(split_number):
            data_number = 'data_' + number_list[i]
            data_number_tra = 'data_' + number_list[i] + '_tra'
            data_number_val = 'data_' + number_list[i] + '_val'
            split_step += split_len
            if i == 0:
                data_split_temp = data_temp[ : split_step]
            elif i == (split_number - 1):
                data_split_temp = data_temp[split_step - split_len:]
            else:
                data_split_temp = data_temp[split_step - split_len: split_step]

            data_split[data_number] += data_split_temp
            # split train and validate dataset and 20% as validate dataset
            tra_val_num = int(np.ceil(len(data_split_temp) * valid_ratio))
            data_split[data_number_tra] += data_split_temp[:tra_val_num]
            data_split[data_number_val] += data_split_temp[tra_val_num:]

    # The directory where the data is stored separately
    for i in range(split_number):
        data_number = 'data_' + number_list[i]
        data_number_tra = 'data_' + number_list[i] + '_tra'
        data_number_val = 'data_' + number_list[i] + '_val'
        data_path = target_path + '/dataset_' + number_list[i] + '.json'
        data_path_tra = target_path + '/dataset_' + number_list[i] + '_tra.json'
        data_path_val = target_path + '/dataset_' + number_list[i] + '_val.json'

        # save first dataset
        with open(data_path, 'w') as f:
            json.dump({'data': data_split[data_number]}, f, indent=1)
        with open(data_path_tra, 'w') as f_t:
            json.dump({'data': data_split[data_number_tra]}, f_t, indent=1)
        with open(data_path_val, 'w') as f_v:
            json.dump({'data': data_split[data_number_val]}, f_v, indent=1)

        print(f'The {data_path} is complete. The length is {len(data_split[data_number])}. ')
        print(f'The {data_path_tra} is complete. The length is {len(data_split[data_number_tra])}. ')
        print(f'The {data_path_val} is complete. The length is {len(data_split[data_number_val])}. ')


def pseudo_soft_target_compute(dataset, pred_result_first, target_first, pred_result_second, target_second, describe):
    with open(dataset, 'r') as f:
        data = pd.DataFrame(json.load(f)['data'])

    with open(pred_result_first, 'r') as f_p_f:
        p_first = pd.DataFrame(csv.DictReader(f_p_f)).astype(float)

    with open(pred_result_second, 'r') as f_p_s:
        p_second = pd.DataFrame(csv.DictReader(f_p_s)).astype(float)

    with open(target_first, 'r') as f_t_f:
        t_first = pd.DataFrame(csv.DictReader(f_t_f)).astype(float)

    with open(target_second, 'r') as f_t_s:
        t_second = pd.DataFrame(csv.DictReader(f_t_s)).astype(float)

    # rename '# 0' to '0'
    p_first.rename(columns={'# 0': 'p0', '1': 'p1', '2': 'p2', '3': 'p3', '4': 'p4', '5': 'p5'}, inplace=True)
    p_second.rename(columns={'# 0': 'p0', '1': 'p1', '2': 'p2', '3': 'p3', '4': 'p4', '5': 'p5'}, inplace=True)
    t_first.rename(columns={'# 0': 't0', '1': 't1', '2': 't2', '3': 't3', '4': 't4', '5': 't5'}, inplace=True)
    t_second.rename(columns={'# 0': 't0', '1': 't1', '2': 't2', '3': 't3', '4': 't4', '5': 't5'}, inplace=True)

    print('audio {} shape is {}'.format(describe, data.shape))
    print('First predict result shape is {}, and first target shape is {}'.format(p_first.shape, t_first.shape))
    print('Second predict result shape is {}, and second target shape is {}'.format(p_second.shape, t_second.shape))
    if p_first.shape == p_second.shape:
        p = (p_first + p_second) / 2.0
    else:
        print('First predict result shape is {} and second predict result shape is{}'.format(p_first.shape,
                                                                                             p_second.shape))
        return

    p_tensor = torch.tensor(p.to_numpy().astype(float))
    t = t_first.to_numpy().astype(float)
    p_argmax, t_argmax = np.argmax(p, 1), np.argmax(t, 1)

    p_max_pd = pd.DataFrame(torch.max(p_tensor, 1).values, columns=['p_max_rate'])
    p_argmax_pd = pd.DataFrame(p_argmax, columns=['p_argmax'])
    t_argmax_pd = pd.DataFrame(t_argmax, columns=['t_argmax'])
    if t_first.shape == t_second.shape:
        data_p_t = pd.concat([data, p, t_first, p_argmax_pd, t_argmax_pd, p_max_pd], axis=1)
    else:
        print('First target shape is {} and second target shape is {}, they are not equal!'.format(t_first.shape,
                                                                                                   t_second.shape))
        return

    return data_p_t


def student_tra_val(data, label_csv, target_path, valid_rate=0.2, temperature=0):
    labels = make_mid_dict(label_csv)

    data = data.sample(frac = 1)
    data_tra = pd.DataFrame()
    data_val = pd.DataFrame()

    # split train and valid
    largest = 0
    for key in labels:
        temp = data[data['labels'] == key]
        labels[key] = len(temp)
        valid_num = int(np.ceil(len(temp) * valid_rate))
        data_tra = pd.concat([data_tra, temp[:-valid_num]], axis=0, ignore_index=True)  # data_tra + temp[:-valid_num]
        data_val = pd.concat([data_val, temp[-valid_num:]], axis=0, ignore_index=True)  # data_val + temp[-valid_num:]
        # get the sample that the class of the largest number
        len_temp = len(temp) - valid_num
        if len_temp > largest:
            largest = len_temp
    print('student training dataset length {}, student validate dataset length {}'.format(len(data_tra), len(data_val)))
    print('student data length {}'.format(len(data_tra) + len(data_val)))

    # sample balancing
    print('\n')
    print('begining student training dataset balancing........')
    balanc_data = pd.DataFrame()
    for key in labels:
        tem_data = data_tra[data_tra['labels'] == key]
        tem_data = tem_data.sample(frac=1)
        balancing_num = largest - len(tem_data)
        print('biggist classes number {} data labels [{}] number {}, need to balancing number {}'
              .format(largest,key , len(tem_data), balancing_num))
        if balancing_num > 0:
            balanc_data = pd.concat([balanc_data, tem_data[:balancing_num]], axis=0, ignore_index=True)

    data_tra = pd.concat([data_tra, balanc_data], axis=0, ignore_index=True)
    print('student training dataset length {}, balancing number {}'.format(len(data_tra), len(balanc_data)))
    print('\n')
    # deal with samples
    data_tra_audio = data_tra.iloc[:, 0:3]
    data_val_audio = data_val.iloc[:, 0:3]
    data_tra_p = data_tra.iloc[:, 3:9]
    data_val_p = data_val.iloc[:, 3:9]
    data_tra_t = data_tra.iloc[:, 9: 15]
    data_val_t = data_val.iloc[:, 9: 15]

    data_tra_json = []
    for i_tra in range(len(data_tra_audio)):
        deal_tra_temp = data_tra_audio.iloc[i_tra]
        tra_json = [{'spk_id':deal_tra_temp['spk_id'], 'wav':deal_tra_temp['wav'], 'labels':deal_tra_temp['labels']}]
        data_tra_json = data_tra_json + tra_json

    data_val_json = []
    for i_val in range(len(data_val_audio)):
        deal_val_temp = data_val_audio.iloc[i_val]
        val_json = [{'spk_id': deal_val_temp['spk_id'], 'wav': deal_val_temp['wav'], 'labels': deal_val_temp['labels']}]
        data_val_json = data_val_json + val_json

    # pseudo soft target temperature in Softmax
    data_tra_p_np = torch.tensor(data_tra_p.to_numpy().astype(float))
    data_val_p_np = torch.tensor(data_val_p.to_numpy().astype(float))
    if temperature != 0:
        print('need to do temperature {}'.format(temperature))
        data_tra_p_np = data_tra_p_np/temperature
        data_tra_p_np = F.softmax(data_tra_p_np, dim=1)
        data_val_p_np = data_val_p_np/temperature
        data_val_p_np = F.softmax(data_val_p_np, dim=1)


    data_tra_t_np = data_tra_t.to_numpy().astype(float)
    data_val_t_np = data_val_t.to_numpy().astype(float)

    # save student dataset
    student_tra_json = target_path + '/student_tra.json'
    student_val_json = target_path + '/student_val.json'
    with open(student_tra_json, 'w') as f_t:
        json.dump({'data': data_tra_json}, f_t, indent=1)
    with open(student_val_json, 'w') as f_v:
        json.dump({'data': data_val_json}, f_v, indent=1)
    print('student training data length is {}, student validate data length is {}'.
          format(len(data_tra_json), len(data_val_json)))

    # save student pseudo soft target
    student_tra_pseudo = target_path + '/student_tra_pseudo.csv'
    student_val_pseudo = target_path + '/student_val_pseudo.csv'
    np.savetxt(student_tra_pseudo, data_tra_p_np, delimiter=',')
    np.savetxt(student_val_pseudo, data_val_p_np, delimiter=',')
    print('student training pseudo soft target shape is {}, student validate pseudo soft target shape is {}'.format(data_tra_p_np.shape, data_val_p_np.shape))

    # save student hard target
    student_tra_hard = target_path + '/student_tra_hard.csv'
    student_val_hard = target_path + '/student_val_hard.csv'
    np.savetxt(student_tra_hard, data_tra_t_np, delimiter=',')
    np.savetxt(student_val_hard, data_val_t_np, delimiter=',')
    print('student training hard target shape is {}, student validate hard target shape is {}'.format(data_tra_t_np.shape, data_val_t_np.shape))
    print('Student dataset create complete!')

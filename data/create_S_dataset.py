import os
import sys
import time
import argparse

import pandas as pd

from data.prep_data import pseudo_soft_target_compute, student_tra_val

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

# split dataset args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--datadir", type=str, default='.',
                    help="dataset path", required=True)
parser.add_argument("--modeldir", type=str, default='/models',
                    help="model path", required=True)
parser.add_argument("--stu_dataset", type=str, default='/datafiles',
                    help="Student dataset", required=True)
parser.add_argument("--t_a_model", nargs='+', default='/baseline_T_A_exp_sizeone,'
                                                         '/baseline_T_A_exp_sizefive',
                    help="the local of the ensemble A model", required=True)
parser.add_argument("--t_b_model", nargs='+', default='/baseline_T_B_exp_sizeone,'
                                                         '/baseline_T_B_exp_sizefive',
                    help="the local of the ensemble B model", required=True)
parser.add_argument("--t_predict", nargs='+', default='/predictions/predictions_dataset_one.csv,'
                                                          '/predictions/predictions_dataset_two.csv',
                    help="the prediction of the ensemble model result which pseudo soft labels", required=True)
parser.add_argument("--target_label", nargs='+', default='/predictions/target_dataset_one.csv,'
                                                         '/predictions/target_dataset_two.csv',
                    help="the target labels", required=True)
parser.add_argument("--t_dataset", nargs='+', default='/split_dataset/dataset_one.json,'
                                                          '/split_dataset/dataset_two.json',
                    help="the ensemble datasets", required=True)

parser.add_argument("--label-csv", type=str, default='/class_labels_indices_vs.csv',
                    help="csv with class labels", required=True)

parser.add_argument("--pred_error", type=bool, default=True,
                    help="filtering prediction error sample", required=True)
parser.add_argument("--filtering_threshold", type=float, default=0.5,
                    help="filtering threshold", required=True)
parser.add_argument("--temperature", type=float, default=1.2,
                    help="temperature in Softmax", required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = args.datadir
    if os.path.exists(data_dir):
        print('The dataset path [ {} ] exists, that is ok!'.format(data_dir))
    else:
        print('The dataset path [ {} ] is not exists, please check your input content!'.format(data_dir))
        sys.exit()

    model_dir = data_dir + args.modeldir

    stu_dataset = data_dir + args.stu_dataset
    if not os.path.exists(stu_dataset):
        os.mkdir(stu_dataset)
        print('create student dataset path [{}]'.format(stu_dataset))
    label_csv = data_dir + args.label_csv
    t_dataset = args.t_dataset.split(',')  # one, two
    t_a_model = args.t_a_model.split(',')  # size_one, size_five depend on model's param
    t_b_model = args.t_b_model.split(',')  # size_one, size_five depend on model's param
    t_predict_result = args.t_predict.split(',')  # one, two
    target_labels = args.target_label.split(',')  # one, two

    # compute the prediction of the ensemble A model by mean
    # 1. get the prediction result of the data two by ensemble model A
    dataset_two = data_dir + t_dataset[1]
    dataset_two_pred_result_first = model_dir + t_a_model[0] + t_predict_result[1]
    dataset_two_target_first = model_dir + t_a_model[0] + target_labels[1]
    dataset_two_pred_result_second = model_dir + t_a_model[1] + t_predict_result[1]
    dataset_two_target_second = model_dir + t_a_model[1] + target_labels[1]
    print('audio dataset two is {}'.format(dataset_two))
    print('the first prediction result of the dataset two is {}'.format(dataset_two_pred_result_first))
    print('the target of the first prediction dataset two is {}'.format(dataset_two_target_first))
    print('the second prediction result of the dataset two is {}'.format(dataset_two_pred_result_second))
    print('the target of the second prediction dataset two is {}'.format(dataset_two_target_second))
    print('Begin to compute the dataset two soft target!')
    # call functions
    data_two_p_t = pseudo_soft_target_compute(dataset_two, dataset_two_pred_result_first, dataset_two_target_first,
                                              dataset_two_pred_result_second, dataset_two_target_second, 'dataset two')

    # 2. get the prediction result of the data one by ensemble model B
    dataset_one = data_dir + t_dataset[0]
    dataset_one_pred_result_first = model_dir + t_b_model[0] + t_predict_result[0]
    dataset_one_target_first = model_dir + t_b_model[0] + target_labels[0]
    dataset_one_pred_result_second = model_dir + t_b_model[1] + t_predict_result[0]
    dataset_one_target_second = model_dir + t_b_model[1] + target_labels[0]
    print('audio dataset one is {}'.format(dataset_one))
    print('the first prediction result of the dataset one is {}'.format(dataset_one_pred_result_first))
    print('the target of the first prediction dataset one is {}'.format(dataset_one_target_first))
    print('the second prediction result of the dataset one is {}'.format(dataset_one_pred_result_second))
    print('the target of the second prediction dataset one is {}'.format(dataset_one_target_second))
    print('Begin to compute the dataset one soft target!')
    # call functions
    data_one_p_t = pseudo_soft_target_compute(dataset_one, dataset_one_pred_result_first, dataset_one_target_first,
                                              dataset_one_pred_result_second, dataset_one_target_second, 'dataset one')

    student_data = pd.concat([data_one_p_t, data_two_p_t], axis=0, ignore_index=True)

    if args.pred_error:
        student_data_fil = student_data[(student_data['p_argmax'] == student_data['t_argmax']) & student_data[
            'p_max_rate'] >= args.filtering_threshold]
    else:
        student_data_fil = student_data[student_data['p_max_rate'] >= args.filtering_threshold]

    print('Filtering prediction error sample is [{}], filtering threshold is [{}]'.format(args.pred_error,
                                                                                          args.filtering_threshold))
    print('source dataset {} \nstudent dataset {} \nfiltering dataset number {}'.format(len(student_data), len(student_data_fil),
                                                                               len(student_data) - len(
                                                                                   student_data_fil)))

    # student data spilt train and validate dataset #valid_ratio
    student_tra_val(student_data_fil, label_csv, stu_dataset, 0.1, args.temperature)


import os
import sys
import time
import argparse
from data.prep_data import alter_data_path, data_set_split


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

# split dataset args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default='.', help="data set", required=True)
parser.add_argument("--data-all", type=str, default='/datafiles/all.json',
                    help="all of dataset", required=True)
parser.add_argument("--label-csv", type=str, default='/class_labels_indices_vs.csv',
                    help="csv with class labels", required=True)
parser.add_argument("--tea-data", type=str, default="/datafiles",
                    help="the location where the split data saved", required=True)
parser.add_argument("--split_number", type=int, default=2,
                    help="The number of data copies that are segmented")
parser.add_argument("--change-path", type=bool, default=False,
                    help="whether need to change dataset's path")

parser.add_argument("--work_path", type=str, default="./",
                    help="the location where the split data saved", required=True)
#args = parser.parse_args(args=[])


if __name__ == '__main__':
    args = parser.parse_args()
    # set work path
    os.chdir(args.work_path)

    data_dir = args.dataset
    if os.path.exists(data_dir):
        print('The ensemble dataset path [ {} ] exists, that is ok!'.format(data_dir))
    else:
        print('The ensemble path [ {} ] is not exists, please check your input content!'.format(data_dir))
        sys.exit()

    data_json_file = data_dir + args.data_all
    class_labels_indices = data_dir + args.label_csv
    split_data_path = data_dir + args.tea_data

    # change the file path of the dataset's json
    if args.change_path:
        alter_data_path(data_dir)

    if not os.path.exists(split_data_path):
        os.mkdir(split_data_path)

    # split all dataset
    # we can define split_rate and split_number to segment dataset to more part in the future,
    # but not now only in half.
    data_set_split(data_json_file, split_data_path, class_labels_indices, 0.2, 0, args.split_number)





import argparse
import os
import pickle
import time
import torch
import shutil
import dataloaders
import models_mulatt
import models_ensemble
import models
from traintest import train, validate
import ast
import numpy as np

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--soft-tralabelcsv", type=str, default='', help="tra csv with pseudo soft class labels")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--eval-dir", type=str, default="", help="evalidation dataset")
# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=60, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=float, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--lrdecay-gamma', default=0.98, type=float, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--n-print-steps", type=int, default=1, help="number of steps to print statistics")
parser.add_argument("--soha-target", type=str, default='hard', help="defined how to train model with hard or soft "
                                                                    "target labels")
parser.add_argument("--con-tra", type=ast.literal_eval, default='False', help="wheter continue training model")
# models args
parser.add_argument("--n_class", type=int, default=17, help="number of classes")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)
parser.add_argument("--model", type=str, default='eff_mean', help="model")
parser.add_argument("--model_size", type=int, default=0, help="model size")
parser.add_argument('--imagenet_pretrain', help='if use pretrained imagenet efficient net', type=ast.literal_eval, default='True')
#
parser.add_argument("--load_model_path", type=str, default=None, help="Load the path of pretrained model")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument('--noise', type=float, default=0, help='add noisy information into original audio')
parser.add_argument('--f-noise', type=float, default=0, help='add noisy information into original audio by fbank')
# evaluate args
parser.add_argument("--eval-dataset", type=str, default='all', help="define which dataset needs to be evaluated")

parser.add_argument("--work_path", type=str, default='', help="define which path you work on")

parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")

args = parser.parse_args()

# set work path
os.chdir(args.work_path)

audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': args.freqm, 'timem': args.timem,
              'mixup': args.mixup, 'noise': args.noise, 'f_noise': args.f_noise, 'mode': 'train'}

print('balanced sampler is not used')
print('Add noisy {} and fbank noisy {} information into original audio'.format(args.noise, args.f_noise))
print(f'Learning rate decay is {args.lr_decay} epoch, learning rate decay gamma is {args.lrdecay_gamma}')
print(f'>>>>>>>>>>>>> Use [ {args.soha_target} ] labels to training model! <<<<<<<<<<<<<<<')
train_loader = torch.utils.data.DataLoader(
    dataloaders.VSDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf,
                          raw_wav_mode=False, specaug=True, soha_target=args.soha_target,
                          soft_label_csv=args.soft_tralabelcsv),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'mixup': 0, 'mode': 'test', 'noise': args.noise, 'f_noise': args.f_noise}

val_loader = torch.utils.data.DataLoader(
    dataloaders.VSDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf,
                          raw_wav_mode=False), 
    batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=True)

print(f'Model is {args.model}......')
if args.model == 'eff_mean':
    audio_model = models.EffNetMean(label_dim=args.n_class, level=args.model_size, pretrain=args.imagenet_pretrain)
elif args.model == 'efficientnet':
    audio_model = models_mulatt.EffNetAttention(label_dim=args.n_class, b=args.model_size, pretrain=args.imagenet_pretrain,
                                         head_num=args.att_head, load_model_path=args.load_model_path)
elif args.model == 'EffNetMeanEnsemble':
    audio_model = models_ensemble.EffNetMeanEnsemble(label_dim=args.n_class, level=[1, 5], pretrain=args.imagenet_pretrain)
else:
    raise ValueError('Model Unrecognized')

# start training
print('Whether continue %s' % args.con_tra)

if not args.con_tra:
    if os.path.exists(args.exp_dir):
        print("Deleting existing experiment directory %s" % args.exp_dir)
        shutil.rmtree(args.exp_dir)
    print("\nCreating experiment directory: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
else:
    print('continue to train model, do not delete %s'%args.exp_dir)

with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = torch.nn.DataParallel(audio_model)

if args.con_tra:
    print('begin to continue training model...........')
    print('loading model {}'.format(args.exp_dir + '/models/best_audio_model.pth'))
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model.load_state_dict(sd)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# test on the test set and sub-test set, model selected on the validation set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('loading model {}'.format(args.exp_dir + '/models/best_audio_model.pth'))
sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)

audio_model.load_state_dict(sd)

all_res = []

# best model on the validation set, repeat to confirm
stats, _, _ = validate(audio_model, val_loader, args, 'valid_set')
# note it is NOT mean of class-wise accuracy
val_acc = stats[0]['acc']
val_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.6f}".format(val_acc))
all_res.append(val_acc)

# test the model on the evaluation set
if args.eval_dataset == 'all':
    data_eval_list = ['te.json', 'subtest/te_age1.json', 'subtest/te_age2.json', 'subtest/te_age3.json', 'subtest/te_female.json', 'subtest/te_male.json']
    eval_name_list = ['all_test', 'test_age_18-25', 'test_age_26-48', 'test_age_49-80', 'test_female', 'test_male']
elif args.eval_dataset == 'two':
    data_eval_list = ['dataset_two.json']
    eval_name_list = ['dataset_two']
else:
    data_eval_list = ['dataset_one.json']
    eval_name_list = ['dataset_one']

eval_dir = args.eval_dir  # '/'.join(data_dir.split('/')[:-1])
for idx, cur_eval in enumerate(data_eval_list):
    cur_eval = eval_dir + '/' + cur_eval
    eval_loader = torch.utils.data.DataLoader(
        dataloaders.VSDataset(cur_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _, predict_output = validate(audio_model, eval_loader, args, eval_name_list[idx])
    eval_acc = stats[0]['acc']
    all_res.append(eval_acc)
    print('---------------evaluate on {:s}---------------'.format(eval_name_list[idx]))
    print("Accuracy: {:.6f}".format(eval_acc))

all_res = np.array(all_res)
all_res = all_res.reshape([1, all_res.shape[0]])
np.savetxt(args.exp_dir + '/all_eval_result.csv', all_res,
           header=','.join(['validation'] + eval_name_list), delimiter=',')


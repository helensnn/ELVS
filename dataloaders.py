import csv
import json
import torchaudio
import numpy as np
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


class VSDataset(Dataset):
    def __init__(self, dataset_json_file, label_csv=None, audio_conf=None, raw_wav_mode=False, specaug=False,
                 soha_target='hard', soft_label_csv=''):
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        self.melbins = self.audio_conf.get('num_mel_bins')
        
        
        self.soha_target = soha_target
        if self.soha_target == 'hard':
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(self.index_dict)
        else:
            with open(soft_label_csv, 'r') as f_soft:
                self.soft_target = pd.DataFrame(csv.reader(f_soft)).to_numpy().astype(float)
            self.label_num = int(self.soft_target.shape[1])
            

        self.windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
                        'bartlett': scipy.signal.bartlett}

        # if just load raw wavform
        self.raw_wav_mode = raw_wav_mode
        if specaug == True:
            self.freqm = self.audio_conf.get('freqm')
            self.timem = self.audio_conf.get('timem')
            
        self.specaug = specaug
        self.mixup = self.audio_conf.get('mixup')
        self.noise = self.audio_conf.get('noise')
        self.fbank_noise = self.audio_conf.get('f_noise')
        

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # do a padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # do cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # define add noise into original audio by fbank
        f_noise_random = random.random()
        if f_noise_random < self.fbank_noise:
            dither_noise = 1.0
        else:
            dither_noise = 0.0
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=dither_noise,
                                                  frame_shift=10)

        target_length = self.audio_conf.get('target_length', 1056)
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):

        if random.random() < self.mixup:
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add labels for the original sample
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add more labels to the original sample labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1 - mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # not doing mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num) + 0.00
            fbank = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                # soft or hard target label
                if self.soha_target == 'hard':
                    label_indices[int(self.index_dict[label_str])] = 1.0
                else:
                    label_indices = self.soft_target[index]

            label_indices = torch.FloatTensor(label_indices)
            mix_lambda = 0

        # add noisy information into original audio
        if random.random() < self.noise:
            fbank = np.array(fbank)
            noise_amp = 0.035*np.random.uniform()*np.amax(fbank)
            fbank = fbank + noise_amp*np.random.normal(size=fbank.shape)
            fbank = torch.FloatTensor(fbank)
            
        # SpecAug, not do for eval set, by intention, it is done after the mix-up step
        if self.specaug == True:
            # Apply masking to a spectrogram in the frequency domain.
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            #
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # mean/std is get from the val set as a prior.
        fbank = (fbank + 3.05) / 5.42

        # shift if in the training set, training set typically use mixup
        if self.mode == 'train':
            fbank = torch.roll(fbank, np.random.randint(0, 1024), 0)
        mix_ratio = min(mix_lambda, 1 - mix_lambda) / max(mix_lambda, 1 - mix_lambda)

        return fbank, label_indices

    def __len__(self):
        return len(self.data)


import os
import json
import random
from pathlib import Path
import logging
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper
import librosa
import torch
import torchaudio
import torchaudio.transforms as AT
import torch.nn.functional as F
from random import randrange

class AihubDataset(torch.utils.data.Dataset):  # type: ignore
    
    _labels = [
        'outer', 'inner', 'ground_transport', 'train_transport', 'water_transport', 'air_transport'
    ]

    def __init__(self, input_dir, dset='', sr=None,
                 resample_rate=None, max_num_targets=1):
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"
        
        self.dset = dset
        self.max_num_targets = max_num_targets
        # local: training_local_total-14_df.pkl
        if dset == 'train':
            self.df_dir = '~/total_data/aihub_noise/Training/training_total-14_df.pkl'
        elif dset == 'val':
            self.df_dir = '~/total_data/aihub_noise/Validation/validation_total-14_df.pkl'
        elif dset == 'test': # TODO
            self.df_dir = '~/total_data/aihub_noise/Validation/validation_total-14_df.pkl'
        
        self.df = pd.read_pickle(self.df_dir)

        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr
    
    def _get_label_vector(self, labels):
        vector = torch.zeros(len(AihubDataset._labels))
        
        for label in labels:
            idx = AihubDataset._labels.index(label)
            assert vector[idx] == 0, 'Repeated'
            vector[idx] = 1

        return vector

    def __len__(self):
        return len(self.df.columns)

    def __getitem__(self, idx):
        col = self.df.iloc[:, idx]
        orig_wav, _ = librosa.load(col['orig_wav_path'], sr=col['sr']) # dim 1
        mixture = torch.from_numpy(np.expand_dims(orig_wav, axis=0))
        noise_wav, _ = librosa.load(col['noise_wav_path'], sr=col['sr'])
        gt = torch.from_numpy(np.expand_dims(noise_wav, axis=0))
        # vocal_wav = librosa.load(col['orig_wav_path'], sr=col['sr'])
        labels = [col['noise_label']]
        label_vector = self._get_label_vector(labels)

        return mixture, label_vector, gt

def collate_fn(batch):
    mlen_mixtures = max([len(x[0][0]) for x in batch])
    mlen_gts = max([len(x[2][0]) for x in batch])
    mixtures = torch.stack([F.pad(x[0], (0, mlen_mixtures - len(x[0][0])), mode='constant', value=0.0) for x in batch])
    labels = torch.stack([x[1] for x in batch])
    gts = torch.stack([F.pad(x[2], (0, mlen_gts - len(x[2][0])), mode='constant', value=0.0) for x in batch])
    return mixtures, labels, gts

def tensorboard_add_sample(writer, tag, sample, step, params):
    if params['resample_rate'] is not None:
        sr = params['resample_rate']
    else:
        sr = params['sr']
    resample_rate = 16000 if sr > 16000 else sr

    m, l, gt, o = sample
    m, gt, o = (
        torchaudio.functional.resample(_, sr, resample_rate).cpu()
        for _ in (m, gt, o))

    def _add_audio(a, audio_tag, axis, plt_title):
        for i, ch in enumerate(a):
            axis.plot(ch, label='mic %d' % i)
            writer.add_audio(
                '%s/mic %d' % (audio_tag, i), ch.unsqueeze(0), step, resample_rate)
        axis.set_title(plt_title)
        axis.legend()

    for b in range(m.shape[0]):
        label = []
        for i in range(len(l[b, :])):
            if l[b, i] == 1:
                label.append(AihubDataset._labels[i])

        # Add waveforms
        rows = 3 # input, output, gt
        fig = plt.figure(figsize=(10, 2 * rows))
        axes = fig.subplots(rows, 1, sharex=True)
        _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
        _add_audio(o[b], '%s/sample_%d/1_output' % (tag, b), axes[1], "Output (%s)" % label)
        _add_audio(gt[b], '%s/sample_%d/2_gt' % (tag, b), axes[2], "GT (%s)" % label)
        writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)

def tensorboard_add_metrics(writer, tag, metrics, label, step):
    """
    Add metrics to tensorboard.
    """
    vals = np.asarray(metrics['scale_invariant_signal_noise_ratio'])

    writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), vals, step)

    label_names = [AihubDataset._labels[torch.argmax(_)] for _ in label]
    for l, v in zip(label_names, vals):
        writer.add_histogram('%s/%s' % (tag, l), v, step)
    


if __name__ == '__main__':
    my_ds = AihubDataset('', dset='train')
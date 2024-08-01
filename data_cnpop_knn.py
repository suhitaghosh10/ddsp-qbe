import os
import random

import torch
from torch.utils.data import Dataset

import librosa
import pyworld as pw


def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False, extension='wav'):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        whole_audio=whole_audio,
        extension=extension,
    is_val=False)

    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        whole_audio=whole_audio,
        extension=extension,
        is_val=True)

    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    return loader_train, loader_valid


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        is_val,
        whole_audio=False,
        extension='wav',
        min_f0=80,
        max_f0=1047.0
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.extension = extension
        self.is_val = is_val
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extension=extension,
            is_pure=True,
            is_sort=True,
            is_ext=False
        )
        self.whole_audio = whole_audio
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.bad_F0 = 5
        self.frame_resolution = (self.hop_size / self.sample_rate)
        self.frame_rate_inv = 1 / self.frame_resolution
        self.frame_period = 1000 * self.frame_resolution

    def __getitem__(self, file_idx
                    ):
        name = self.paths[file_idx]

        # check duration. if too short, then skip
        duration = librosa.get_duration(
            path=os.path.join(self.path_root, 'audio', name) + '.'+self.extension,
            sr=self.sample_rate)

        if duration < (self.waveform_sec + 0.1):
            return self.__getitem__(file_idx+1)

        # get item
        return self.get_data(name, duration)

    def get_data(self, name, duration):
        # path
        path_audio = os.path.join(self.path_root, 'audio', name) + '.'+self.extension
        audio_wavlm6_ = os.path.join(self.path_root, 'wavlm', name) + '.pt'
        audio_wavlm12_ = os.path.join(self.path_root, 'wavlm12', name) + '.pt'

        # load audio
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        idx_from = 0 if self.whole_audio or self.is_val else random.uniform(0, duration - waveform_sec - 0.1)

        audio, sr = librosa.load(
                path_audio,
                sr=self.sample_rate,
                offset=idx_from,
                duration=waveform_sec)
        if self.sample_rate != sr:
            librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # clip audio into N seconds
        audio = audio[...,:audio.shape[-1]//self.hop_size*self.hop_size]

        wvlm_frame_len = int(waveform_sec*self.frame_rate_inv)
        strt = int(idx_from * self.frame_rate_inv)

        audio_wavlm6_ = torch.load(audio_wavlm6_, map_location='cpu',weights_only=True).float()
        audio_wavlm6 = audio_wavlm6_[strt:strt+wvlm_frame_len]

        audio_wavlm12_ = torch.load(audio_wavlm12_, map_location='cpu',weights_only=True).float()
        audio_wavlm12 = audio_wavlm12_[strt:strt + wvlm_frame_len]

        x = audio.astype('double')
        f0, t = pw.dio(
            x,
            self.sample_rate,
            f0_floor=65.0,
            f0_ceil=1047.0,
            channels_in_octave=2,
            frame_period=(1000 * self.frame_resolution))
        f0 = pw.stonemask(x, f0, t, self.sample_rate)
        f0 = f0.astype('float')[:audio_wavlm6.size(0)]
        f0_hz = torch.from_numpy(f0).float().unsqueeze(-1)
        f0_hz[f0_hz < self.min_f0] *= 0
        f0_normalised = self.__znorm__(f0_hz, False)

        audio = torch.from_numpy(audio).float()
        assert sr == self.sample_rate
        return dict(audio=audio, f0=f0_hz, norm_f0=f0_normalised, mel=audio_wavlm6, mel12=audio_wavlm12, name=name)

    def __len__(self):
        return len(self.paths)

    def __znorm__(self, x, log=False):
        if x.max() == 0:
            return x
        else:
            if log:
                x = torch.where(x<1,0.,torch.log10(x))
            return (x - x.mean()) / torch.sqrt(x.var())
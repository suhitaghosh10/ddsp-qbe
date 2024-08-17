from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
import torchaudio.transforms as T

from wavlm.WavLM import WavLM

from utils import get_F0, CUDA


class PhoneMapper(nn.Module):

    def __init__(self,
                 wavlm: WavLM,
                 sr: int,
                 hop_length: int,
                 device: str | None = CUDA
                 ) -> None:
        """ Phone Mapper module: Produces the mapped phonetic representations and the prosodic features from the source.
        """
        super().__init__()
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = sr
        self.hop_length = hop_length
        self.wavlm_dim2 = 1024
        self.phone_layer = 6
        self.prosody_layer = 12

    @torch.inference_mode()
    def get_phone_pool(self, wavs: list[Path], vad_trigger_level: int = 7) -> Tensor:
        """ Get wavlm features for the provided waveforms, needed to create the pool for matching.
        """
        x6_pool = []
        for wav in wavs:
            feats = self.get_features_from_source(wav, vad_trigger_level=vad_trigger_level, return_emo=False,
                                                  return_f0=False)
            x6_pool.append(feats['x6'])
        x6_pool = torch.concat(x6_pool, dim=0).cpu()
        return x6_pool

    @torch.inference_mode()
    def get_features_from_source(self, wav_path: str, vad_trigger_level: int = 7, hop_size: int = 320,
                                 return_emo: bool = False, return_f0: bool = False):
        """Returns x6 x12 f0 features of the provided waveform,
        optionally perform VAD trimming on start/end.
        """
        # load audio
        x, sr = torchaudio.load(wav_path, normalize=True)

        if not sr == self.sr:
            print(f"resample {sr} to {self.sr} in {wav_path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr

        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            x = waveform_end_trim

        wav_input_16khz = x.to(self.device)
        inps = {
            'x6': self.wavlm.extract_features(wav_input_16khz, output_layer=self.phone_layer, ret_layer_results=False)[
                0].squeeze(0)}

        if return_emo:
            inps['x12'] = self.wavlm.extract_features(wav_input_16khz, output_layer=self.prosody_layer,
                                                      ret_layer_results=False)[0].squeeze(0)

        if return_f0:
            x = x[-1].numpy().astype('double')
            _, f0_norm = get_F0(x, sr, hop_size, min_f0=80., normalised=True)
            inps['f0'] = f0_norm.unsqueeze(0)[:, :inps['x6'].size(0), :]

        return inps

    @torch.inference_mode()
    def get_mapping(self, src: str, pool: Tensor, topn: int = 10, device: str | None = 'cpu') -> Tensor:
        """ Given `src` waveform and `pool` of x6 tensors, find the closest phonetic (x6) representations of `src`.
        Inputs:
            - `src`: str path of the source wav.
            - `pool`: Tensor (N, dim) of the pool for a target speaker.
            - `topn`: n selected (best matching) candidates to source x6.
            - `device`: if None, uses default device at initialization. Otherwise uses specified device
        Returns:
            - converted the mapped x6, x12_src and normalised F0 tensors
        """
        device = torch.device(device) if device is not None else self.device
        # get the inputs to be fed to model
        inp_dict = self.get_features_from_source(src, return_emo=True, return_f0=True)
        pool = pool.to(device)
        x6_src = inp_dict['x6'].to(device)
        x12_src = inp_dict['x12'].to(device)
        f0_src = inp_dict['f0'].to(device)

        cosine_dists = self._cosine_dist(x6_src, pool, device=device)
        best_val, best_ind = cosine_dists.topk(k=topn, largest=False, dim=-1)
        weights = torch.softmax(1 / best_val, dim=-1)
        best_mapped_x6 = pool[best_ind]
        weights_reshaped = weights.unsqueeze(1).repeat(1, 1, self.wavlm_dim2).reshape(best_mapped_x6.shape[0],
                                                                                      topn,
                                                                                      self.wavlm_dim2)
        best_mapped_x6 = best_mapped_x6 * weights_reshaped
        best_mapped_x6 = best_mapped_x6.mean(dim=1)

        return best_mapped_x6, x12_src, f0_src

    def _cosine_dist(self, source_feats: Tensor, matching_pool: Tensor, device: str | None = 'cpu') -> Tensor:
        """ taken from knn-vc. Like torch.cdist, but fixed dim=-1 and for cosine distance."""
        source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
        matching_norms = torch.norm(matching_pool, p=2, dim=-1)
        dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0] ** 2 + source_norms[:,
                                                                                                  None] ** 2 + \
                  matching_norms[None] ** 2
        dotprod /= 2
        dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
        return dists

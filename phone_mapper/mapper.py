from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
import torchaudio.transforms as T
from fusion_synthesis.ddsp.vocoder import SubtractiveSynthesiser

from wavlm.WavLM import WavLM

from utils import get_F0
from phone_mapper.mapper_utils import cosine_dist, WAVLM_EMBEDDING_DIM_2, SPEAKER_INFORMATION_LAYER, EMO_INFORMATION_LAYER


class PhoneMapper(nn.Module):

    def __init__(self,
                 wavlm: WavLM,
                 ddsp: SubtractiveSynthesiser,
                 sr: int,
                 hop_length: int,
                 device='cuda'
                 ) -> None:
        """ Phone Mapper.
        """
        super().__init__()
        self.ddsp = ddsp.eval()
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = sr
        self.hop_length = hop_length

    @torch.inference_mode()
    def get_phone_pool(self, wavs: list[Path], vad_trigger_level=7) -> Tensor:
        """ Get wavlm features for the provided waveforms, needed to create the pool for matching.
        """
        x6_pool = []
        for wav in wavs:
            feats = self.get_inputs(wav, vad_trigger_level=vad_trigger_level, return_emo=False, return_f0=False)
            x6_pool.append(feats['x6'])
        x6_pool = torch.concat(x6_pool, dim=0).cpu()
        return x6_pool


    @torch.inference_mode()
    def get_inputs(self, wav_path: str, vad_trigger_level=0, hop_size=320, return_emo=False, return_f0=False):
        """Returns x6 x12 normalised f0 representations of the provided waveform as a tensor of shape (seq_len, dim),
        optionally perform VAD trimming on start/end.
        """
        # load audio
        x, sr = torchaudio.load(wav_path, normalize=True)
                
        if not sr == self.sr :
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
        inps = {}

        speaker_features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
        inps['x6'] = speaker_features.squeeze(0)

        if return_emo:
            emo_features = self.wavlm.extract_features(wav_input_16khz, output_layer=EMO_INFORMATION_LAYER,
                                                           ret_layer_results=False)[0]
            inps['x12'] = emo_features.squeeze(0)

        if return_f0:
            x = x[-1].numpy().astype('double')
            _, f0_norm = get_F0(x, sr, hop_size, min_f0=80, normalised=True)
            inps['f0'] = f0_norm.unsqueeze(0)[:,:emo_features.size(1),:]

        return inps





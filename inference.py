
import torch
from phone_mapper.mapper_utils import get_wavlm
from phone_mapper.mapper import PhoneMapper
from fusion_synthesis.ddsp.vocoder import SubtractiveSynthesiser
import glob
import os
import soundfile as sd
from torch import Tensor
import torchaudio



@torch.inference_mode()
def synthesise(ddsp: SubtractiveSynthesiser, x6: Tensor, x12: Tensor, f0: Tensor, tgt_loudness_db: float | None = -16, device: str | None = None) -> Tensor:
        """ Given `src` waveform and `pool` x6 tensors, find closest phonetic (x6) representations of `src`.
        Inputs:
            - `src`: str path of the source wav.
            - `pool`: Tensor (N, dim) of the pool for a target speaker.
             - `topn`: n number of candidates chosen from the pool.
            - `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable.
            - `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
            - `device`: if None, uses default device at initialization. Otherwise uses specified device
        Returns:
            - converted waveform of shape (T,)
        """

        prediction,  _, _, _, _ = ddsp(x6=x6[None].to(device), x12=x12[None].to(device),
                                                      f0_norm=f0.to(device))

        prediction = prediction.squeeze().cpu()

        # normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], sr)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)

        else:
            pred_wav = prediction
        return pred_wav

if __name__ == '__main__':
    BLACKMAN_HARRIS_WINDOW = 'blackman-harris'
    device = 'cpu'
    hop_length = 320
    sr = 16000
    ref_num = 100
    N = 10

    root_path = '/cache/sghosh/exp/'
    chkpt = 'ckpts/vocoder_best_params.pt'
    device = torch.device(device)
    model = SubtractiveSynthesiser(
        sampling_rate=16000,
        block_size=320,
        n_mag_harmonic=176,
        n_mag_noise=80,
        n_harmonics=150,
        device=device
    )
    # load ddsp model's weights
    model.load_state_dict(torch.load('/scratch/sghosh/sad.pt', map_location=device))
    model.to(device)
    model.eval()
    wavlm = get_wavlm(True, True, device)
    mapper = PhoneMapper(wavlm=wavlm, device=device, hop_length=hop_length, sr=sr)


    r_path = '/scratch/sghosh/datasets/DDSP/libri_esd/train/audio/8063/'
    src_wav_path = '/scratch/sghosh/datasets/librispeech-ddsp/dev-clean-orig/audio/7976/110124/7976-110124-0018.flac'
    ref_wav_paths = glob.glob(os.path.join(r_path, '*/*.flac'))[0:ref_num]
    print(ref_wav_paths[0:10])
    # get pool
    pool = mapper.get_phone_pool(ref_wav_paths)
    x6, x12, f0_norm = mapper.get_mapping(src=src_wav_path, pool=pool)
    pred_wav = synthesise(ddsp=model, x6=x6, x12=x12, f0=f0_norm, device='cpu')
    sd.write('/scratch/sghosh/exp/1.wav', data=pred_wav, samplerate=16000)
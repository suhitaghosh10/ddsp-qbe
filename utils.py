import pyworld as pw
from parselmouth.praat import call
import parselmouth
import torch
import logging

from wavlm.WavLM import WavLM, WavLMConfig

HANN: str = 'hann'

BLACKMAN: str = 'blackman'

BLACKMAN_HARRIS: str = 'blackman-harris'

CUDA: str = 'cuda'


def get_F0(signal, sr, hop_size, min_f0=80.):
    frame_resolution = (hop_size / sr)
    f0, _ = pw.dio(
        signal,
        sr,
        f0_floor=65.0,
        f0_ceil=1047.0,
        channels_in_octave=2,
        frame_period=(1000 * frame_resolution))

    f0_hz = torch.from_numpy(f0.astype('float32')).unsqueeze(-1)
    f0_hz[f0_hz < min_f0] *= 0
    f0_normalised = znorm(f0_hz, False)
    return f0_hz, f0_normalised


def znorm(x, log: bool=False):
    if x.max() == 0:
        return x
    else:
        if log:
            x = torch.where(x < 1, 0., torch.log10(x))
        return (x - x.mean()) / torch.sqrt(x.var())


def get_hnr(wav, min_pitch: float=80., sr: int=16000):
    sound = parselmouth.Sound(wav, sr)
    harmonic = call(sound, "To Harmonicity (cc)", 0.01, min_pitch, 0.1, 1.0)
    hnr = call(harmonic, "Get mean", 0, 0)
    return hnr


def get_wavlm(pretrained=True, progress=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See
    https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    if not torch.cuda.is_available():
        if str(device) != 'cpu':
            logging.warning(f"No GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/suhitaghosh10/ddsp-qbe/blob/main/resources/model_weights/WavLM-Large.pt",
        map_location=device,
        progress=progress
    )

    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model
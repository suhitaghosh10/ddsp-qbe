import torch
import torch.nn as nn

from .controlgenerator import ControlGen
from .modules import WaveGeneratorOscillator
from .utils import scale_function, frequency_filter, upsample, unit_to_F0

class SubtractiveSynthesiser(nn.Module):
    def __init__(self,
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_mels=1024,
            channel_num_filter=64,
            window_type='hann',
            convolve_power=1,
            is_odd=True,
            device='cuda',):
        super().__init__()

        print(' [Model] Sawtooth (with sinusoids) Subtractive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.window_type = window_type
        self.is_odd = is_odd
        self.convolve_power = convolve_power
        self.channel_num_filter=channel_num_filter
        self.hz_min = torch.tensor(80.0, device=device)
        self.hz_max = torch.tensor(1000.0, device=device)

        # Mel2Control
        split_map = {
            'f0': 1,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = ControlGen(n_mels, split_map, num_channels=channel_num_filter)

        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, n_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        self.harmonic_synthsizer = WaveGeneratorOscillator(
            sampling_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

    def forward(self, mel, mel12, f0_norm, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel, mel12, f0_norm)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_F0(f0_unit, f0_min=self.hz_min, f0_max=self.hz_max, use_log=True)
        f0 = torch.where(f0 < 80, 0., f0)

        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = f0.shape

        # upsample
        pitch = upsample(f0, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param,
                        window_type=self.window_type,
                        is_odd=self.is_odd,
                        convolve_power=self.convolve_power)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param,
                        window_type=self.window_type,
                        is_odd=self.is_odd,
                        convolve_power=self.convolve_power)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)

import torch
import torch.nn as nn

from .controlgenerator import ControlGen
from .modules import SawtoothOscillator
from .dsp import scale_function, frequency_filter, upsample, unit_to_F0

class SubtractiveSynthesiser(nn.Module):
    def __init__(self,
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_wavlm=1024,
            channel_num_filter=64,
            window_type='hann',
            convolve_power=1,
            is_odd=True,
            min_f0=80.,
            max_f0=1000.0,
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
        self.hz_min = torch.tensor(min_f0, device=device)
        self.hz_max = torch.tensor(max_f0, device=device)
        self.min_f0 = min_f0

        # Mel2Control
        split_map = {
            'f0': 1,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.embedding_to_controls = ControlGen(n_wavlm, split_map, num_channels=channel_num_filter)

        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, n_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        self.harmonic_synthesizer = SawtoothOscillator(
            sampling_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

    def forward(self, x6, x12, f0_norm, x6_emo1=None, x6_emo2=None, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls, x_phon_emo1, x_phon_emo2 = self.embedding_to_controls(x6, x12, f0_norm=f0_norm, x6_emo1=x6_emo1, x6_emo2=x6_emo2)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_F0(f0_unit, f0_min=self.hz_min, f0_max=self.hz_max, use_log=True)
        f0 = torch.where(f0 < self.min_f0, 0., f0)

        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = f0.shape

        # upsample
        pitch = upsample(f0, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
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

        return signal, f0, final_phase, (harmonic, noise), (x_phon_emo1, x_phon_emo2)

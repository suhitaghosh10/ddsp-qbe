import torch
import torch.nn as nn
import numpy as np

from .utils import linear_lookup, upsample, remove_above_nyquist


def safe_division(a, b, eps=1e-9):
    mask = (b <= eps)
    c = a / b
    c[mask] = 0.0
    return c


class HarmonicOscillator(nn.Module):
    """synthesize audio with a bank of harmonic oscillators"""

    def __init__(self, fs, oscillator=torch.sin, is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.oscillator = oscillator
        self.is_remove_above_nyquist = is_remove_above_nyquist

    def forward(self, f0, amplitudes, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
            amplitudes: B x T x n_harmonic
         initial_phase: B x 1 x 1
          ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)

        mask = (f0 > 0).detach()
        f0 = f0.detach()

        # harmonic synth
        n_harmonic = amplitudes.shape[-1]
        phase = torch.cumsum(2 * np.pi * f0 / self.fs, axis=1) + initial_phase
        phases = phase * torch.arange(1, n_harmonic + 1).to(phase)

        # anti-aliasing
        if self.is_remove_above_nyquist:
            amp = remove_above_nyquist(amplitudes, f0, self.fs)
        else:
            amp = amplitudes.to(phase)

        # signal
        signal = (self.oscillator(phases) * amp).sum(-1, keepdim=True)
        signal *= mask
        signal = signal.squeeze(-1)

        # phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)
        return signal, final_phase.detach()


class WaveGeneratorOscillator(nn.Module):
    """
        synthesize audio with a sawtooth oscillator.
        the sawtooth oscillator is synthesized by a bank of sinusoids
    """

    def __init__(
            self,
            fs,
            amplitudes,
            ratio,
            is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.is_remove_above_nyquist = is_remove_above_nyquist
        self.amplitudes = amplitudes
        self.ratio = ratio
        self.n_harmonics = len(amplitudes)

    def forward(self, f0, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
         initial_phase: B x 1 x 1
          ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)

        mask = (f0 > 0).detach()
        f0 = f0.detach()

        # harmonic synth
        phase = torch.cumsum(2 * np.pi * f0 / self.fs, axis=1) + initial_phase
        phases = phase * torch.arange(1, self.n_harmonics + 1).to(phase)

        # anti-aliasing
        amplitudes = self.amplitudes * self.ratio
        if self.is_remove_above_nyquist:
            amp = remove_above_nyquist(amplitudes.to(phase), f0, self.fs)
        else:
            amp = amplitudes.to(phase)

        # signal
        signal = (torch.sin(phases) * amp).sum(-1, keepdim=True)
        signal *= mask
        signal = signal.squeeze(-1)

        # phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)

        return signal, final_phase.detach()


class SawtoothGenerator(nn.Module):
    """synthesize audio with a sawtooth oscillator"""

    def __init__(self, fs, is_reversed=True):
        super().__init__()
        self.fs = fs
        self.is_reversed = is_reversed

    def forward(self, f0, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
            amplitudes: B x T x 1
         initial_phase: B x 1 x 1
           ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)

        mask = (f0 > 0).detach()
        f0 = f0.detach()

        # sawtooth
        phase = torch.cumsum(2 * np.pi * f0 / self.fs, axis=1) + initial_phase
        if self.is_reversed:
            phase = 2 * np.pi - phase % (2 * np.pi)
        else:
            phase = phase % (2 * np.pi)
        signal = (phase / np.pi) - 1
        signal *= mask
        signal = signal.squeeze(-1)

        final_phase = phase[:, -1:, :]
        return signal, final_phase.detach()
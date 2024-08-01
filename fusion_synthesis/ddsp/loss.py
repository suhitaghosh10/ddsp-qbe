import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import librosa

MIN_F0 = 50

class PerceptualLoss(nn.Module):
    def __init__(self, n_ffts, jitter_weight=10, shimmer_weight=0.1, relative_jitter=True, use_kurtosis=False):
        super().__init__()

        self.loss_mss_func = MSSLoss(n_ffts, use_kurtosis=use_kurtosis)
        self.f0_loss_func = F0L1Loss()
        self.jitter_loss = JitterLoss(relative=relative_jitter)
        self.jitter_weight = jitter_weight
        self.shimmer_weight = shimmer_weight
        self.shimmer_loss = ShimmerLoss()
        self.use_kurtosis = use_kurtosis

    def forward(self, y_pred, y_true, f0_pred, f0_true, is_val):
        loss_mss, loss_kurtosis = self.loss_mss_func(y_pred, y_true)
        loss_f0 = self.f0_loss_func(f0_pred, f0_true, is_val)
        loss_jitter = self.jitter_loss(f0_pred, f0_true, is_val)
        loss_shimmer = self.shimmer_loss(y_pred, y_true, is_val)

        loss = loss_mss + loss_f0 + \
               int(self.use_kurtosis == True) * loss_kurtosis + \
               self.jitter_weight * loss_jitter + \
               self.shimmer_weight * loss_shimmer

        return loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer)


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7, use_kurtosis=False, sr=16000):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.use_kurtosis = use_kurtosis
        if use_kurtosis:
            self.spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sr, n_fft=self.n_fft,
                                                                            hop_length=self.hop_length).to('cuda')
            self.freqs = torch.from_numpy(librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)).to('cuda').unsqueeze(
                0).unsqueeze(0)

    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term

        if self.use_kurtosis:
            pred_kurtosis = self._get_kurtosis(S_pred, x_pred)
            real_kurtosis = self._get_kurtosis(S_true, x_true)

            pred_kurtosis = torch.nan_to_num(pred_kurtosis)
            real_kurtosis = torch.nan_to_num(real_kurtosis)

            nan_count = torch.isnan(pred_kurtosis).sum() + torch.isnan(real_kurtosis).sum()
            kurtosis_loss = torch.nan_to_num(
                F.smooth_l1_loss(real_kurtosis, pred_kurtosis, reduction='sum') / (real_kurtosis.numel() - nan_count),
                nan=0)
            return {'loss': loss, 'k_loss': kurtosis_loss}
        else:
            return {'loss': loss, 'k_loss': torch.tensor(0., device='cuda')}

    def _get_kurtosis(self, S, x):

        SC = self.spectral_centroid(x).unsqueeze(-1)
        spectrogram = torch.abs(S).transpose(-2, -1)
        pred_spread = torch.sqrt(
            torch.sum((self.freqs - SC) ** 2 * spectrogram, dim=-1, keepdim=True) / torch.sum(
                spectrogram,
                dim=-1,
                keepdim=True) + 10e-3)
        kurtosis = torch.sqrt(
            torch.sum((self.freqs - SC) ** 4 * spectrogram, dim=-1, keepdim=True) / (
                    pred_spread ** 4 * torch.sum(SC, dim=-1, keepdim=True)) + 10e-3)
        return kurtosis



class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio=1.0, overlap=0.75, eps=1e-7, use_kurtosis=True):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps, use_kurtosis=use_kurtosis) for n_fft in n_ffts])
        self.ratio = ratio

    def forward(self, x_pred, x_true, return_spectrogram=True):
        x_pred = x_pred[..., :x_true.shape[-1]]
        if return_spectrogram:
            losses = []
            k_losses = []
            for loss in self.losses:
                loss_dict = loss(x_true, x_pred)
                losses += [loss_dict['loss']]
                k_losses += [loss_dict['k_loss']]

        return self.ratio * sum(losses).sum(), sum(k_losses).sum()


class F0L1Loss(nn.Module):
    """
    F0 linear and log loss
    """

    def __init__(self):
        super().__init__()
        self.iteration = 0

    def forward(self, f0_predict, f0_hz_true, is_val):
        if not is_val:
            self.iteration += 1
        #print(is_val, self.iteration)

        if (len(f0_hz_true.size()) != 3):
            f0_hz_true = f0_hz_true.unsqueeze(-1)

        if torch.sum(f0_hz_true) == 0:
            return torch.tensor(0.0, device='cuda')
        if self.iteration > 5000:
            f0_predict = torch.where(f0_hz_true < MIN_F0, f0_predict * 0.0, f0_predict)
            loss = F.l1_loss(torch.log(f0_hz_true + 1e-3), torch.log(f0_predict + 1e-3), reduction='sum')
            loss = loss / torch.sum(f0_hz_true >= MIN_F0)
        else:
            loss = F.l1_loss(torch.log(f0_hz_true + 1e-3), torch.log(f0_predict + 1e-3), reduction='mean')

        return torch.sum(loss)


class JitterLoss(nn.Module):
    """
    jitter loss
    """

    def __init__(self, relative):
        super().__init__()
        self.iteration = 0
        self.eps = 1e-3
        self.relative = relative

    def forward(self, f0_predict, f0_hz_true, is_val):

        if not is_val:
            self.iteration += 1

        if (len(f0_hz_true.size()) != 3):
            f0_hz_true = f0_hz_true.unsqueeze(-1)

        if torch.sum(f0_hz_true) == 0:
            return torch.tensor(0.0, device='cuda')

        if self.iteration > 5000:
            f0_predict = torch.where(f0_hz_true < 50, f0_predict * 0.0, f0_predict)
            jitter_pred, jitter_true = self._get_jitter_loss(f0_hz_true, f0_predict, self.relative)
            loss = F.l1_loss(jitter_pred, jitter_true, reduction='sum') / torch.sum(f0_hz_true >= 50)
        else:
            jitter_pred, jitter_true = self._get_jitter_loss(f0_hz_true, f0_predict, self.relative)
            loss = F.l1_loss(jitter_true, jitter_pred, reduction='mean')
        return torch.sum(loss)

    def _get_jitter_loss(self, f0_hz_true, f0_predict, relative):
        f0_true_period = torch.where(f0_hz_true == 0., self.eps, 1 / (f0_hz_true + self.eps))
        jitter_true_abs = ((torch.abs(f0_true_period[:, :-1] - f0_true_period[:, 1:])).mean(1))
        jitter_true = jitter_true_abs / (f0_true_period.mean(1)) if relative else jitter_true_abs

        f0_pred_period = torch.where(f0_predict == 0., self.eps, 1 / (f0_predict + self.eps))
        jitter_pred_abs = ((torch.abs(f0_pred_period[:, :-1] - f0_pred_period[:, 1:])).mean(1))
        jitter_pred = jitter_pred_abs / (f0_pred_period.mean(1)) if relative else jitter_pred_abs

        return jitter_true, jitter_pred


class ShimmerLoss(nn.Module):
    """
    loss based on absolute shimmer
    """

    def __init__(self, name='ShimmerLoss'):
        super().__init__()
        self.iteration = 0
        self.eps = 1e-3

    def forward(self, x_pred, x_true, is_val):
        if not is_val:
            self.iteration += 1

        x_pred = x_pred[..., :x_true.shape[-1]]
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        if self.iteration > 5000:
            shimmer_pred, shimmer_true = self._get_shimmer_loss(x_pred, x_true)
            if torch.isnan(shimmer_true) or torch.isnan(shimmer_pred) or torch.isinf(shimmer_true) or torch.isinf(
                    shimmer_pred):
                loss = 0.
            else:
                loss = F.l1_loss(shimmer_pred, shimmer_true, reduction='mean')
                loss = torch.tensor(0., device='cuda') if torch.isnan(loss) or torch.isinf(loss) else loss
            return loss
        else:
            return torch.tensor(0.0, device='cuda')

    def _datacheck_peakdetect(self, x_axis, y_axis):
        if x_axis is None:
            x_axis = range(len(y_axis))

        if len(y_axis) != len(x_axis):
            raise ValueError(
                "Input vectors y_axis and x_axis must have same length")

        # needs to be a numpy array
        y_axis = torch.asarray(y_axis)
        x_axis = torch.asarray(x_axis)
        return x_axis, y_axis

    def _peak_detect(self, x, window_size=100):
        windowed_power = torch.square(x)  # Square the signal
        max_pool_output = torch.nn.functional.max_pool1d_with_indices(windowed_power, window_size, window_size)[0]
        min_pool_output = torch.abs(
            torch.nn.functional.max_pool1d_with_indices(-windowed_power, window_size, window_size)[0])
        mean_pool_output = torch.nn.functional.avg_pool1d(windowed_power, window_size, window_size)[0]

        return max_pool_output, min_pool_output, mean_pool_output

    def _get_shimmer_loss(self, x_true, x_pred, eps=1e-7):
        max_amplitude_t, min_amplitude_t, mean_amplitude_t = self._peak_detect(x_true)
        max_amplitude_p, min_amplitude_p, mean_amplitude_p = self._peak_detect(x_pred)
        # relative shimmer
        shimmer_true = torch.mean(10 * torch.log10((max_amplitude_t - min_amplitude_t) + eps / mean_amplitude_t + eps))
        shimmer_pred = torch.mean(10 * torch.log10((max_amplitude_p - min_amplitude_p) + eps / mean_amplitude_p + eps))
        return shimmer_pred, shimmer_true
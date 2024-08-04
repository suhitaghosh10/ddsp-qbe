import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np
from typing import Text
from scipy import fftpack


def logb(x, base=2.0):
  """Logarithm with base as an argument."""
  return torch.log(x) / torch.log(torch.tensor(base))


def unit_to_F0(unit,
               f0_min,
               f0_max,
               use_log=True) :
    """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
    if use_log:
        f0_max = torch.log(f0_max)
        f0_min = torch.log(f0_min)
        return torch.exp(((f0_max - f0_min) * unit) + f0_min)
    else:
        return ((f0_max - f0_min) * unit) + f0_min

def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-7
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(np.log(10)) + 1e-7


def fft_convolve(audio,
                 impulse_response,
                 padding = 'same',
                 delay_compensation = -1):
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().
    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    #audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

    # Add a frame dimension to impulse response if it doesn't have one.
    #ir_shape = impulse_response.shape.as_list()
    ir_shape = impulse_response.size()
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        #ir_shape = impulse_response.shape.as_list()
        ir_shape = impulse_response.size()

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size()#shape.as_list()

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                        'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size

    # audio --> batch, time_step
    audio = audio.unsqueeze(1)
    if frame_size!=audio_size:
        filters = torch.eye(frame_size).unsqueeze(1).cuda()
        audio_frames = F.conv1d(audio, filters, stride= hop_size).transpose(1,2)
        n_audio_frams = audio_frames.size(1)
    else:
        audio_frames = audio#.transpose(1,2)
    # Check that number of frames match.
    #print ("audio_frames here:", audio_frames.size())
    n_audio_frames = int(audio_frames.shape[1])

    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)

    audio_fft = torch.fft.rfft(audio_frames, fft_size)

    ir_fft = torch.fft.rfft(impulse_response, fft_size)
    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    #audio_out = torch.signal.overlap_and_add(audio_frames_out, hop_size)
    if frame_size!=audio_size:
        overlap_add_filter = torch.eye(audio_frames_out.size(-1), requires_grad = False).unsqueeze(1).cuda()
        #print (overlap_add_filter.size())
        #print (audio_frames_out.size(), overlap_add_filter.size())
        output_signal = F.conv_transpose1d(audio_frames_out.transpose(1, 2),
                                                        overlap_add_filter,
                                                        stride = frame_size,
                                                        padding = 0).squeeze(1)
    else:
        output_signal = crop_and_compensate_delay(audio_frames_out.squeeze(1), audio_size, ir_size, padding,
                                    delay_compensation)
    return output_signal[...,:frame_size*n_ir_frames]
    # Crop and shift the output audio.

def linear_lookup(index,
                  wavetable):
    '''
    Args:
        index: B x T, value: [0, 1]
        wavetables: B x num_wavetables x len_wavetable
    Returns:
        singal: B x num_wavetables x T
    '''
    wavetable = torch.cat([wavetable, wavetable[..., 0:1]], dim=-1)
    wavetable = wavetable.unsqueeze(-1)        # B x C x L x 1

    index = index.unsqueeze(-1) * 2 - 1        # [0, 1] -> [-1, 1]
    index_ = torch.zeros_like(index)
    index = torch.cat([index_, index], dim=-1) # B x L x 2
    index = index.unsqueeze(2)                 # B x L x 1 x 2

    signal = F.grid_sample(
        wavetable,
        grid=index,
        mode='bilinear',
        align_corners=True).squeeze(-1)
    return signal

def crop_and_compensate_delay(audio:torch.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> torch.Tensor:
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]

def apply_window_to_impulse_response(impulse_response,
                                     window_size: int = 0,
                                     causal: bool = False,
                                     window_type='hann',
                                     convolve_power=1,
                                     is_odd=True):
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????

        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """


    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    #ir_size = int(impulse_response.shape[-1])
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size

    # if the size of the window is even, then making it odd so that it's centered around 0, and we get to do analysis for phase 0
    if is_odd and window_size % 2 == 0:
        window_size -= 1

    window = nn.Parameter(get_window(type=window_type, convolve_power=convolve_power, window_width=window_size),
                          requires_grad = False).cuda()
    # Zero pad the window and put in in zero-phase form.

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding], device='cuda'),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll((window.size(-1)+1)//2, -1)

    # Apply the window, to get new IR (both in zero-phase form).

    window = window.unsqueeze(0)
    impulse_response = impulse_response * window

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                    impulse_response[..., :second_half_end]],
                                    dim=-1)
    else:
        impulse_response = impulse_response.roll((impulse_response.size(-1)+1)//2, -1)

    return impulse_response

def frequency_impulse_response(magnitudes,
                               window_type='hann',
                               convolve_power=1):
    """Get windowed impulse responses using the frequency sampling method.
    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html
    Args:
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
    Returns:
        impulse_response: Time-domain FIR filter of shape
        [batch, frames, window_size] or [batch, window_size].
    Raises:
        ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).

    magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes)

    #print ("impulse response size here:", impulse_response[0,0, :5], impulse_response[0,0, -5:])

    """ This means this?
    First: Initilize at fourier space, where real part equal magnitueds, complex part equal 0
    First: Initilize at fourier space, where real part equal magnitueds, complex part equal 0
    Second: Convert back to time domain
    """

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response,
                                                        impulse_response.size(-1),
                                                        window_type=window_type,
                                                        is_odd=True,
                                                        convolve_power=convolve_power)
    return impulse_response


def frequency_filter(audio,
                     magnitudes,
                     padding = 'same',
                     mel_scale_noise = False,
                     window_type= 'hann',
                     convolve_power=1,
                    is_odd=True
                     ):
    """Filter audio with a finite impulse response filter.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it is set as the default (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        window_size - 1).
    Returns:
        Filtered audio. Tensor of shape
            [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    """

    impulse_response = frequency_impulse_response(magnitudes,
                                                  window_type=window_type,
                                                  convolve_power=convolve_power)

    return fft_convolve(audio, impulse_response, padding=padding)

def blackman_harris_window(n, dtype=torch.float32, device='cuda'):
    if n <= 1:
        return torch.tensor([1.0], dtype=dtype)

    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    # Create a linear space from 0 to n-1
    t = torch.linspace(0, n - 1, n, dtype=dtype, device=device)

    # Calculate the window values using the Blackman-Harris formula
    window = a0 - a1 * torch.cos(2 * math.pi * t / (n - 1)) + a2 * torch.cos(4 * math.pi * t / (n - 1)) - a3 * torch.cos(6 * math.pi * t / (n - 1))

    return window

def get_window(type, convolve_power, window_width, device='cuda'):
    if convolve_power < 1: raise Exception('please provide the power >=1 for blackman-harris window')
    else:
        if type == 'hann':
            return torch.pow(torch.hann_window(window_width, device=device),convolve_power)
        elif type == 'blackman':
            return torch.pow(torch.blackman_window(window_width, device=device),convolve_power)
        elif type == 'blackman-harris':
            return torch.pow(blackman_harris_window(window_width, device=device),convolve_power)
        else:
            raise Exception('please provide a valid window type: hann, blackman or blackman-harris')

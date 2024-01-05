import logging
import os
import sys
import time
from datetime import timedelta

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from torch.distributions.categorical import Categorical

import librosa
import torchaudio
import torchaudio.transforms as T

from pathlib import Path
from scipy.io import wavfile

from distutils.version import LooseVersion
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")

def generate_waveform(cfg, outputs) -> torch.Tensor:
    if cfg.trainer.mol.active:
        m = torch.distributions.Uniform(0, 1)
        logit_probs_mol = outputs.transpose(1, 2)[:, :, :cfg.trainer.mol.n_mols]
        means = outputs[:, cfg.trainer.mol.n_mols:cfg.trainer.mol.n_mols * 2, :]  # bs, channels, timesteps
        log_scales = torch.clamp(
            outputs[:, cfg.trainer.mol.n_mols*2:cfg.trainer.mol.n_mols * 3, :], min=-7.0
        )  # bs, channels, timesteps
        if torch.isnan(log_scales).sum():
            print("log_scales", log_scales)
        models_chosen = torch.nn.functional.gumbel_softmax(logit_probs_mol, tau=0.01, hard=True, eps=1e-7, dim=-1)
        if torch.isnan(models_chosen).sum():
            print("models_chosen", models_chosen)
        # Categorical do no work for gradient, approximate with gumbel_softmax
        chosen_scales = (log_scales.transpose(1, 2) * models_chosen).sum(-1).unsqueeze(1)
        chosen_means = (means.transpose(1, 2) * models_chosen).sum(-1).unsqueeze(1)
        samples_uniform = m.sample([outputs.size(0), 1, outputs.size(2)]).cuda()  # bsz, value, timesteps
        if torch.isnan(samples_uniform).sum():
            print("samples_uniform", samples_uniform)
        waveform_generated = chosen_means + torch.exp(chosen_scales) * samples_uniform
        if torch.isnan(waveform_generated).sum():
            print("waveform_generated_0", waveform_generated)

    elif cfg.trainer.classes > 1:
        if cfg.trainer.sample_max:
            probabilities_outputs = torch.softmax(outputs, 1)
            _, indices = torch.max(probabilities_outputs, 1)
        else:
            probabilities_outputs = torch.softmax(outputs, 1)
            categorical_distrib = Categorical(probabilities_outputs.transpose(1, 2))
            indices = categorical_distrib.sample()

        waveform_generated = mu_law_decoding(indices, quantization_channels=cfg.trainer.classes)
        # LOG.info("The waveform generated doesn't track gradient yet, to be implemented.")
    else:
        waveform_generated = outputs
        
    return waveform_generated

def snr(estimate, reference, power = 2, epsilon=1e-8):
    '''
    Calculate Signal-to-Noise ratio
    '''
    noise = reference - estimate
    noise_power = noise.norm(p = power)
    reference_power = reference.norm(p = power)
    snr = reference_power / noise_power
    snr = 20*torch.log10(snr)
    return snr


def si_sdr(estimate, reference, epsilon=1e-8):
    '''
    Calculate the Scale-invariant signal-to-distortion ration.
    cf. SDR - half-baked or well done? , https://arxiv.org/abs/1811.02508
    '''

    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)
    sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return sisdr

def make_melspectrogram_list(multiple_stft = True, sample_rate = 44100, stable = True, log_scale = True):
    win_length = None
    mel_spectrogram_list = []
    if multiple_stft:
        n_ffts = [1024,2048,4096,1024, 2048, 4096]
        hop_lengths = [512, 512, 256, 128, 256, 256]
        n_mels = [64, 128, 256, 128, 128, 256]              
        for n_fft, hop_length, n_mel in zip(n_ffts, hop_lengths, n_mels):
            if stable:
                mel_spectrogram = MelSpectrogram_Stable(sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=None,
                    window="hann",
                    n_mels=n_mel,
                    fmin=60,
                    fmax=22050,
                    center=True,
                    normalized=True,
                    pad_mode="reflect",
                    onesided=True,
                    eps=1e-6,
                    log_base=10.0,
                    log_scale=log_scale
                ).cuda()
            else:
                mel_spectrogram = T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                    norm="slaney",
                    onesided=True,
                    n_mels=n_mel,
                    mel_scale="htk",
            ).cuda()
            mel_spectrogram_list.append(mel_spectrogram)
    else:
        n_fft = 1024
        hop_length = 512
        n_mels = 128
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        ).cuda()
        mel_spectrogram_list.append(mel_spectrogram)
    return mel_spectrogram_list

class MelSpectrogram_Stable(torch.nn.Module):
    """Calculate Mel-spectrogram."""

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        win_length=None,
        window="hann",
        n_mels=128,
        fmin=80,
        fmax=22050,
        center=True,
        normalized=False,
        pad_mode="reflect",
        onesided=True,
        eps=1e-5,
        log_base=10.0,
        log_scale=False
    ):
        """Initialize MelSpectrogram module."""
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.pad_mode = pad_mode
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window
        self.eps = eps
        self.log_scale = log_scale
        fmin = 0 if fmin is None else fmin
        fmax = sample_rate / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        ##Not to be updated pass the params to the buffer
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())
        self.stft_params = {
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "pad_mode":self.pad_mode
        }
        if  is_pytorch_17plus:
            self.stft_params["return_complex"] = False

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f"log_base: {log_base} is not supported.")

    def forward(self, x):
        """Calculate Mel-spectrogram.
        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, 1, T).
        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).
        """
        if torch.isnan(x).sum():
            print("x", x, x.dtype, x.size())
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=x.dtype, device=x.device)
        else:
            window = None
        x_stft = torch.stft(x, window=window, **self.stft_params)

        # (B, #freqs, #frames, 2) -> (B, $frames, #freqs, 2), the last 2 dims are real and imaginary parts
        x_stft = x_stft.transpose(1, 2)
        if torch.isnan(x_stft).sum():
            print('x_stft', x_stft, x_stft.dtype)
            print('x',x, x.size(), x.dtype)
        x_power = x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2
        if torch.isnan(x_power).sum():
            print('x_power', x_power)
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps))
        if torch.isnan(x_amp).sum():
            print('x_amp', x_amp)
            raise RuntimeError
        #x_amp = torch.sqrt(x_power)
        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)
        
        if self.log_scale :
            return self.log(x_mel).transpose(1, 2)
        else:
            return x_mel.transpose(1,2)

def spectral_loss(waveform_generated, target_waveform, mel_spectrogram_list = None, scale = False, loss= 'l2', use_db = True):
    power_to_db = T.AmplitudeToDB()
    power_to_db.top_db = 100.0
    power_to_db.amin = 1e-7
    melspec_loss = 0
    melspec_loss_per_stft = []
    for melspec in mel_spectrogram_list:
        if use_db:
            melspectrogram_generated = power_to_db(melspec(waveform_generated.squeeze(1)))
            melspectrogram_target = power_to_db(melspec(target_waveform.float().squeeze(1)))
        else:
            melspectrogram_generated = melspec(waveform_generated.squeeze(1))
            melspectrogram_target = melspec(target_waveform.float().squeeze(1))
        if loss=='l2':
            melspec_loss_temp = F.mse_loss(melspectrogram_generated,melspectrogram_target)
        if loss=='l1':
            melspec_loss_temp = F.l1_loss(melspectrogram_generated,melspectrogram_target)
        if loss=='mix':
            melspec_loss_l2 = F.mse_loss(melspectrogram_generated,melspectrogram_target)
            melspec_loss_l1 = F.l1_loss(melspectrogram_generated,melspectrogram_target)
            melspec_loss_temp = (melspec_loss_l1 +melspec_loss_l2)/2

        if scale:                      
            melspec_loss += torch.div(melspec_loss_temp, torch.abs(melspectrogram_target)).mean()
        else:
            melspec_loss += melspec_loss_temp.mean() 
        melspec_loss_per_stft.append(melspec_loss.item())

    return melspec_loss / len(mel_spectrogram_list), melspec_loss_per_stft


def cross_entropy_loss(input, target):
    # input:  (batch, 256, len)
    # target: (batch, len)

    batch, channel, seq = input.size()

    input = input.transpose(1, 2).contiguous()
    input = input.view(-1, 256)  # (batch * seq, 256)
    target = target.view(-1).long()  # (batch * seq)

    cross_entropy = F.cross_entropy(input, target, reduction="none")  # (batch * seq)
    return cross_entropy.reshape(batch, seq).mean(dim=1)  # (batch)


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=2048, log_scale_min=-6.0, reduce=True):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].
    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    # If X ~ U(0, 1) then μ + s(log(X) − log(1 − X)) ~ Logistic(μ, s)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    ### makes sure target is as needed
    if y.dim() == 2:
        y = y.unsqueeze(2)
    if y.dim() == 3:
        if y.size(1) == 1:  ### means channel is second
            y = y.transpose(1, 2)

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix : 3 * nr_mix], min=log_scale_min)
    
    if torch.isnan(log_scales).sum():
        print("log_scales", log_scales)
    if torch.isnan(logit_probs).sum():
        print("logit_probs", logit_probs)
    if torch.isnan(means).sum():
        print("means", means)
    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    if torch.isnan(inv_stdv).sum():
        print("inv_stdv", inv_stdv)

    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    if torch.isnan(cdf_plus).sum():
        print("cdf_plus", cdf_plus)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)
    if torch.isnan(cdf_min).sum():
        print("cdf_min", cdf_min)
    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in+1e-5)
    # log_cdf_plus = torch.log(torch.sigmoid(plus_in) + 1e-5)
    if torch.isnan(log_cdf_plus).sum():
        print("log_cdf_plus", log_cdf_plus)
    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in+1e-5)
    # log_one_minus_cdf_min = torch.log(1 - torch.sigmoid(min_in) + 1e-5)
    if torch.isnan(log_one_minus_cdf_min).sum():
        print("log_one_minus_cdf_min", log_one_minus_cdf_min)
    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in+1e-5)
    # log_pdf_mid = mid_in - log_scales + 2 * torch.log(1 - torch.sigmoid(min_in) + 1e-5)
    if torch.isnan(log_pdf_mid).sum():
        print("log_pdf_mid", log_pdf_mid)
    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()
    if torch.isnan(inner_inner_cond).sum():
        print("inner_inner_cond", inner_inner_cond)

    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-6)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - np.log((num_classes - 1) / 2)
    )
    if torch.isnan(inner_inner_out).sum():
        print("inner_inner_out", inner_inner_out)
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    if torch.isnan(inner_out).sum():
        print("inner_out", inner_out)

    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    if torch.isnan(log_probs).sum():
        print("log_probs", log_probs)
    log_probs = log_probs + F.log_softmax(logit_probs + 1e-6, -1)
    if torch.isnan(log_probs).sum():
        print("log_probs2", log_probs)
    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


class timeit:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is None:
            print(f"{self.name} took {(time.time() - self.start) * 1000} ms")
        else:
            self.logger.debug("%s took %s ms", self.name, (time.time() - self.start) * 1000)


def linear_pcm(x, classes=256):
    return (x / (2 / classes)).astype("int16") + (classes / 2)


def cut_track_stack(audio_input_tensor, window_length=8192*2**4, overlap=0.5):
    """
    Cuts a given track in overlapping windows and stacks them along a new axis
    :param track_path: path to .wav track to apply the function on
    :param window_length: number of samples per window (scalar int)
    :param overlap: ratio of overlapping samples for consecutive samples (scalar int in [0, 1))
    :return: processed track as a numpy array with dimension [window_number, 1, window_length], sampling frequency
    """
    track = audio_input_tensor
    # Get number of windows and prepare empty array
    window_number = compute_window_number(track_length=track.size(-1), window_length=window_length,overlap=overlap)
    bsz = track.size(0)
    #cut_track = np.zeros((window_number, window_length))
    cut_track = torch.zeros((bsz,window_number, window_length))
    # Cut the tracks in smaller windows
    for music_index in range(bsz):
        for i in range(window_number):
            window_start = int(i * (1 - overlap) * window_length)
            window = track[:,window_start: window_start + window_length]

            # Check if last window needs padding
            if window.size(-1) != window_length:
                padding = window_length - window.size(1)
                window = torch.cat([window, torch.zeros([bsz,padding])], axis = 1)
                # padding = window_length - window.shape[0]
                # window = np.concatenate([window, np.zeros(padding)])

            cut_track[music_index,i] = window
    return cut_track


def compute_window_number(track_length: int, window_length: int = 8192*2**5, overlap: float = 0.5):
    """
    Computes the number of overlapping window for a specific track.
    :param track_length: total number of samples in the track (scalar int).
    :param window_length: number of samples per window (scalar int).
    :param overlap: ratio of overlapping samples for consecutive samples (scalar int in [0, 1))
    :return: number of windows in the track
    """
    num = track_length - window_length
    den = window_length * (1 - overlap)
    return int(num // den + 2)

def overlap_and_add_samples(samples_tensor: torch.Tensor, overlap: float, window_length: int, use_windowing: bool = True) -> torch.Tensor:
    """
    Re-construct a full sample from its sub-parts using the OLA algorithm.
    :param batch: input signal previously split in overlapping windows torch tensor of shape [B, 1, WINDOW_LENGTH].
    :return: reconstructed sample (torch tensor).
    """
    # Compute the size of the full sample
    bsz, window_number, single_sample_size = samples_tensor.size()
    full_sample_size = int(single_sample_size * (1 + (window_number - 1) * (1 - overlap)))

    # Initialize the full sample
    full = torch.zeros((bsz, full_sample_size))

    if use_windowing:
        hanning = torch.from_numpy(np.hanning(window_length))

    for batch in range(bsz):
        for window_index in range(window_number):
            window_start = int(window_index * (1 - overlap) * window_length)
            window_end = window_start + window_length
           
            sample = samples_tensor[batch, window_index].squeeze()
            if use_windowing:               
                sample *= hanning

            full[batch,window_start: window_end] += sample
        return full


def plot_signal(signal, title, path, cmap="hot"):
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use("agg")

    fig = plt.figure()
    if signal.ndim == 1:
        plt.plot(signal)
    else:
        plt.imshow(signal, cmap=cmap, aspect="auto")
    plt.title(title)
    plt.savefig(path)
    plt.close(fig)

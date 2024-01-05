from typing import Any, Tuple

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import torch


def plot_waveform(
    waveform: Any,
    sample_rate: int = 44100,
    max_points: int = 22050,
    title: str = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    ax: Any = None,
    **kwargs
) -> librosa.display.AdaptiveWaveplot:

    # https://librosa.org/doc/latest/generated/librosa.display.waveshow.html#librosa.display.waveshow
    w = waveform
    if type(w) is torch.Tensor:
        w = waveform.T[:, 0].numpy()

    if not ax:
        fig, ax = plt.subplots()

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set(title=title)
    ax.label_outer()

    return librosa.display.waveshow(w, sr=sample_rate, ax=ax, **kwargs)


def compare_waveforms(
    left: Any,
    sample_rate: int,
    right: Any,
    max_points: int = 22050,
    title: str = None,
    label_left: str = None,
    label_right: str = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    ax: Any = None,
    **kwargs
) -> librosa.display.AdaptiveWaveplot:

    if not ax:
        fig, ax = plt.subplots()

    plot_waveform(
        left,
        sample_rate,
        max_points,
        title,
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        label=label_left,
        alpha=0.5,
        color="c",
        **kwargs
    )
    plot_waveform(
        right,
        sample_rate,
        max_points,
        title,
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        label=label_right,
        alpha=0.5,
        color="r",
        **kwargs
    )
    ax.legend()

    return ax

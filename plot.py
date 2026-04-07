import matplotlib.pyplot as plt
import numpy as np
from dsp_signal import Signal

def generate_plot(ax, signal: Signal) -> None:
    ax.clear()
    ax.plot(signal.time, signal.signal)
    ax.set(xlabel='Time [s]', ylabel='Amplitude', title='Signal plot')
    ax.grid(True)

def generate_discrete_plot(ax, signal: Signal) -> None:
    ax.clear()

    # If signal is full of 0'os then stem plot breaks
    if len(signal.signal) == 0 or all(v == 0 for v in signal.signal):
        ax.text(
            0.5, 0.5,
            "Discrete signal contains no impulses",
            ha='center', va='center', fontsize=12
        )
        ax.set_axis_off()
        return
    
    markerline, stemlines, baseline = ax.stem(
        signal.time, signal.signal, linefmt='b-', markerfmt='bo', basefmt='r-'
    )
    plt.setp(stemlines, 'linewidth', 0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title('Discrete Signal Plot')
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_histogram(ax, signal, bins=5, title="Signal histogram"):
    ax.clear()

    values = signal.signal
    counts, bin_edges, patches = ax.hist(
        values,
        bins=bins,
        color="mediumseagreen",
        edgecolor="black"
    )

    # Show all bin edges as ticks
    ax.set_xticks(bin_edges)
    ax.tick_params(axis='x', rotation=45)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

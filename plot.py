import matplotlib.pyplot as plt
import numpy as np

def generate_plot(signal: "Signal") -> None:
    t = np.arange(signal.start_time, signal.duration + signal.start_time, 1 / signal.sample_rate)
    s = signal.signal.copy()

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='A',
           title='Signal plot')
    ax.grid()

    plt.show()

def generate_discrete_plot(signal: "Signal") -> None:
    t = np.arange(signal.start_time, signal.duration + signal.start_time, 1 / signal.sample_rate)
    s = signal.signal.copy()

    plt.figure(figsize=(10, 4))
    markerline, stemlines, baseline = plt.stem(t, signal.signal, linefmt='b-', markerfmt='bo', basefmt='r-')

    plt.setp(stemlines, 'linewidth', 0.1)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Discrete Signal Plot')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()


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



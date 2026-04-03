import signal
import plot

if __name__ == "__main__":
    print("Hello World")
    test_signal = signal.generate_continuous_signal(10, 5.5, 0, 1,
                                                    signal.SignalType.TRIANGULAR, 0.6)

    test_discrete_signal = signal.generate_discrete_signal(10, 5.5, 0, 1,
                                                           signal.SignalType.IMPULSE_NOISE, 0.1)
    print(test_signal)
    plot.generate_plot(test_signal)
    plot.generate_discrete_plot(test_discrete_signal)
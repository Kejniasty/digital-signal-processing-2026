import signal
import plot

if __name__ == "__main__":
    print("Hello World")
    test_signal = signal.generate_continuous_signal(10, 5.5, 0, 1,
                                                    signal.SignalType.TRIANGULAR, 0.6)

    print(test_signal)
    plot.generate_plot(test_signal)
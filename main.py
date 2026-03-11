import signal
import plot

if __name__ == "__main__":
    print("Hello World")
    test_signal = signal.generate_continuous_signal(10, 5.5, 0, 1,
                                                    signal.SignalType.RECT_SYMMETRIC, 0.3)

    print(test_signal)
    plot.generate_plot(test_signal)
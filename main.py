import dsp_signal as signal
import plot
import file_operations as file

if __name__ == "__main__":
    print("Hello World")
    test_signal = signal.generate_continuous_signal(10, 5.5, 0, 1,
                                                    signal.SignalType.TRIANGULAR, 0.6)

    test_discrete_signal = signal.generate_discrete_signal(10, 5.5, 0, 1,
                                                           signal.SignalType.IMPULSE_NOISE, 0.1)
    print(test_signal)
    plot.generate_plot(test_signal)
    plot.generate_discrete_plot(test_discrete_signal)
    plot.plot_histogram(test_signal.signal, bins=10, title="Histogram of the signal")

    filename = "test.txt"
    file.signal_to_file(test_signal, filename)
    test_signal = file.signal_from_file(filename)
    print(test_signal)
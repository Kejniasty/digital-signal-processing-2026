import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtWidgets import QFileDialog

import plot
from dsp_signal import *
from file_operations import *
from mpl_canvas import MplCanvas   # dodasz ten plik

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        loader = QUiLoader()
        ui_file = QFile("gui/main.ui")
        ui_file.open(QFile.ReadOnly)

        loaded_ui = loader.load(ui_file)
        ui_file.close()

        if loaded_ui is None:
            print("UI failed to load!")
        else:
            print("UI loaded successfully")

        self.setCentralWidget(loaded_ui)
        self.ui = loaded_ui

        # Matplotlib canvas inside "plots"
        self.canvas = MplCanvas(self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.plots.setLayout(layout)

        # Connect UI elements
        self.ui.GenerateSignal.clicked.connect(self.generate_signal)
        self.ui.ShowHistogram.clicked.connect(self.show_histogram)
        self.ui.saveSig.clicked.connect(self.save_signal)
        self.ui.loadSig.clicked.connect(self.load_signal)
        self.ui.bins.valueChanged.connect(self.update_bins_label)

        self.signal = None  # placeholder

        # signal types for plots
        self.discrete_types = {
            SignalType.DIRAC_DELTA,
            SignalType.IMPULSE_NOISE
        }


    

    def update_bins_label(self, value):
        snapped = round(value / 5) * 5
        self.ui.bins.blockSignals(True)
        self.ui.bins.setValue(snapped)
        self.ui.bins.blockSignals(False)
        self.ui.binLabel.setText(f"bins: {snapped}")

    def generate_signal(self):
        amp = self.ui.amplitude.value()
        freq = self.ui.frequency.value()
        dur = self.ui.duration.value()
        start_time = self.ui.startTime.value()
        coefficient = self.ui.coeff.value()
        sample_rate = self.ui.sampleRate.value()

        sig_type = list(SignalType)[self.ui.typeDropdown.currentIndex()]

        period = 1 / freq if freq != 0 else 1

        # --- DISCRET SIGNALS ---
        if sig_type in (SignalType.DIRAC_DELTA, SignalType.IMPULSE_NOISE):
            self.signal = generate_discrete_signal(
                amplitude=amp,
                duration=dur,
                start_time=start_time,
                period=period,
                type=sig_type,
                coefficient=coefficient,
                sample_rate=sample_rate
            )

        # --- CONTINOUS SIGNALS ---
        else:
            self.signal = generate_continuous_signal(
                amplitude=amp,
                duration=dur,
                start_time=start_time,
                period=period,
                type=sig_type,
                coefficient=coefficient,
                sample_rate=sample_rate
            )

        self.plot_signal()

    def plot_signal(self):
        if self.signal is None:
            return

        sig_type = list(SignalType)[self.ui.typeDropdown.currentIndex()]
        is_discrete = sig_type in self.discrete_types

        if is_discrete:
            plot.generate_discrete_plot(self.canvas.ax, self.signal)
        else:
            plot.generate_plot(self.canvas.ax, self.signal)

        self.canvas.draw()

    def show_histogram(self):
        if self.signal is None:
            return

        bins = self.ui.bins.value()
        plot.plot_histogram(self.canvas.ax, self.signal, bins=bins)
        self.canvas.draw()


    def save_signal(self):
        if self.signal is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save signal",
            "",
            "Binary signal (*.bin)"
        )

        if filename:
            signal_to_file(self.signal, filename)


    def load_signal(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load signal",
            "",
            "Binary signal (*.bin)"
        )

        if filename:
            self.signal = signal_from_file(filename)
            self.plot_signal()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

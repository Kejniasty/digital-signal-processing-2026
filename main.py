import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtWidgets import QFileDialog

import plot
from dsp_signal import *
from file_operations import *
from mpl_canvas import MplCanvas

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
        self.showing_histogram = False


        # Connect UI elements
        self.ui.GenerateSignal.clicked.connect(self.generate_signal)
        self.ui.ShowHistogram.clicked.connect(self.show_histogram)
        self.ui.saveSig.clicked.connect(self.save_signal)
        self.ui.loadSig.clicked.connect(self.load_signal)
        self.ui.bins.valueChanged.connect(self.update_bins_label)
        self.ui.savePlot.clicked.connect(self.savePlot)
        self.ui.addBtn.clicked.connect(self.op_add)
        self.ui.subBtn.clicked.connect(self.op_sub)
        self.ui.multBtn.clicked.connect(self.op_mul)
        self.ui.divBtn.clicked.connect(self.op_div)
        self.ui.sampleSignal.clicked.connect(self.op_sample)
        self.ui.quantize.clicked.connect(self.op_quantize)
        self.ui.reconstruct.clicked.connect(self.op_reconstruct)
        self.ui.metrics.clicked.connect(self.op_metrics)
        self.ui.aliasing.clicked.connect(self.op_aliasing)


        self.signal = None  # placeholder
        self.loaded_signals = {}  # key = filename, value = Signal object

        self.type_map_ui_to_enum = {
            "Uniform": "UNIFORM_NOISE",
            "Gaussian": "GAUSSIAN_NOISE",
            "Sine": "SINE",
            "Half wave rectangular sine": "HALF_WAVE_RECT_SINE",
            "Full wave sine": "FULL_WAVE_RECT_SINE",
            "Rectangular": "RECT",
            "Rectangular symmetric": "RECT_SYMMETRIC",
            "Triangular": "TRIANGULAR",
            "Heavy side step": "HEAVISIDE_STEP",
            "Dirac delta": "DIRAC_DELTA",
            "Impulse Noise": "IMPULSE_NOISE"
        }

        self.type_map_enum_to_ui = {v: k for k, v in self.type_map_ui_to_enum.items()}

        # signal types for plots
        self.discrete_types = {
            SignalType.DIRAC_DELTA,
            SignalType.IMPULSE_NOISE
        }
    # --- BUTTON HOOKUPS ---

    # oveall button functionality section
    def savePlot(self):
        if self.signal is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF File (*.pdf)"
        )

        if filename:
            self.canvas.figure.tight_layout()
            self.canvas.figure.savefig(filename)

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

        # --- DISCRETE SIGNALS ---
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

        # --- CONTINUOUS SIGNALS ---
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
        self.original_signal = self.signal
        self.plot_signal()

    def plot_signal(self):
        if self.signal is None:
            return

        if hasattr(self.signal, "is_aliasing") and self.signal.is_aliasing:
            plot.generate_plot(self.canvas.ax, self.signal)
            self.canvas.draw()
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

        sig_type = list(SignalType)[self.ui.typeDropdown.currentIndex()]
       
        if self.showing_histogram:
            self.ui.ShowHistogram.setText("Show histogram")
        else:
            self.ui.ShowHistogram.setText("Show signal")

        # if hist go back to plot
        if self.showing_histogram:
            self.canvas.ax.clear()

            # chart type choose
            if sig_type in (SignalType.DIRAC_DELTA, SignalType.IMPULSE_NOISE):
                plot.generate_discrete_plot(self.canvas.ax, self.signal)
            else:
                plot.generate_plot(self.canvas.ax, self.signal)

            self.canvas.draw()
            self.showing_histogram = False
            return

        # if plot show hist
        plot.plot_histogram(
            self.canvas.ax,
            self.signal,
            bins=self.ui.bins.value(),
            title="Signal histogram"
        )

        self.canvas.draw()
        self.showing_histogram = True


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
            ui_text = self.ui.typeDropdown.currentText()
            enum_name = self.type_map_ui_to_enum[ui_text]

            signal_to_file(self.signal, filename, enum_name)

    def load_signal(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load signal",
            "",
            "Binary signal (*.bin)"
        )

        if filename:
            loaded_signal, enum_name = signal_from_file(filename)
            self.signal = loaded_signal

            ui_text = self.type_map_enum_to_ui[enum_name]

            index = self.ui.typeDropdown.findText(ui_text)
            if index != -1:
                self.ui.typeDropdown.setCurrentIndex(index)

            self.loaded_signals[filename] = self.signal
            self.refresh_signal_dropdowns()

            self.plot_signal()

    def pretty_name(self, path: str) -> str:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        return name

    def refresh_signal_dropdowns(self):
        self.ui.firstSig.clear()
        self.ui.secSig.clear()

        for full_path in self.loaded_signals.keys():
            display_name = self.pretty_name(full_path)
            self.ui.firstSig.addItem(display_name, full_path)
            self.ui.secSig.addItem(display_name, full_path)
    
    def get_selected_signals(self):
        full1 = self.ui.firstSig.currentData()
        full2 = self.ui.secSig.currentData()

        if full1 not in self.loaded_signals or full2 not in self.loaded_signals:
            return None, None

        return self.loaded_signals[full1], self.loaded_signals[full2]

    # specific button section
    def op_add(self):
        sig1, sig2 = self.get_selected_signals()
        if sig1 is None:
            return

        result = sig1 + sig2
        self.signal = result
        self.plot_signal()

    def op_sub(self):
        sig1, sig2 = self.get_selected_signals()
        if sig1 is None:
            return

        result = sig1 - sig2
        self.signal = result
        self.plot_signal()

    def op_mul(self):
        sig1, sig2 = self.get_selected_signals()
        if sig1 is None:
            return

        result = sig1 * sig2
        self.signal = result
        self.plot_signal()

    def op_div(self):
        sig1, sig2 = self.get_selected_signals()
        if sig1 is None:
            return

        result = sig1 / sig2
        self.signal = result
        self.plot_signal()
        
    def op_sample(self):
        if self.signal is None:
            return

        new_sr = self.ui.sampleRate.value()

        if new_sr == self.signal.sample_rate:
            print("Sampling skipped: new sample rate equals original sample rate.")
            return


        if new_sr > self.signal.sample_rate:
            print("Cannot sample to a higher sample rate — use reconstruction instead.")
            return

        try:
            self.signal = self.signal.sample(new_sr)
        except Exception as e:
            print("Sampling error:", e)
            return

        self.plot_signal()

    def op_quantize(self):
        if self.signal is None:
            return

        levels = self.ui.quantLevels.value()
        method = self.ui.quantMethod.currentText()

        if method == "Mid-tread (round)":
            self.signal = self.signal.quantize_mid_tread(levels)
        else:
            self.signal = self.signal.quantize_mid_rise(levels)

        self.plot_signal()

    def op_reconstruct(self):
        if self.signal is None:
            return

        target_sr = self.ui.sampleRate.value()
        method = self.ui.reconMethod.currentText()

        if method == "ZOH":
            self.signal = self.signal.reconstruct_zoh(target_sr)
        elif method == "FOH":
            self.signal = self.signal.reconstruct_foh(target_sr)
        else:
            self.signal = self.signal.reconstruct_sinc(target_sr)

        self.plot_signal()

    def op_metrics(self):
        if self.signal is None:
            return
        if not hasattr(self, "original_signal") or self.original_signal is None:
            print("No original signal stored.")
            return

        orig = self.original_signal
        proc = self.signal

        if self.original_signal is None:
            print("No original signal stored.")
            return

        self.ui.mseLabel.setText(f"MSE: {mse(orig, proc):.6f}")
        self.ui.snrLabel.setText(f"SNR: {snr(orig, proc):.2f} dB")
        self.ui.psnrLabel.setText(f"PSNR: {psnr(orig, proc):.2f} dB")
        self.ui.mdLabel.setText(f"MD: {md(orig, proc):.6f}")

    def op_aliasing(self):
        fo = self.ui.foInput.value()
        fs = self.ui.fsInput.value()
        amp_interf = self.ui.ampInterfInput.value()

        if fo <= 0 or fs <= 0:
            print("Invalid fo or fs")
            return

        duration = 1.0
        fa = fo + fs
        CONT_SR = max(50000, 10 * fa)

        # continous time
        t = np.arange(0, duration, 1/CONT_SR)

        useful = np.sin(2 * np.pi * fo * t)

        interfer = amp_interf * np.sin(2 * np.pi * fa * t)

        # sum
        combined = useful + interfer

        # sampling
        t_new = np.arange(0, duration, 1/fs)
        x_new = np.interp(t_new, t, combined)

        sampled = Signal(
            signal=x_new.tolist(),
            amplitude=max(abs(x) for x in x_new),
            duration=duration,
            start_time=0.0,
            period=1.0,
            sample_rate=fs
        )

        sampled.is_aliasing = True

        self.signal = sampled
        self.original_signal = sampled
        self.plot_signal()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
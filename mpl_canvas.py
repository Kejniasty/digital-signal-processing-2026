from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6, 8))
        super().__init__(self.figure)

        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

        self.set_single_mode()

    def clear_figure(self):
        self.figure.clear()

    def set_single_mode(self):
        self.clear_figure()

        self.ax1 = self.figure.add_subplot(111)
        self.ax2 = None
        self.ax3 = None

        self.figure.tight_layout(pad=3.0)

    def set_double_mode(self):
        self.clear_figure()

        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.ax3 = None

        self.figure.tight_layout(pad=3.0)

    def set_triple_mode(self):
        self.clear_figure()

        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(313)

        self.figure.tight_layout(pad=3.0)
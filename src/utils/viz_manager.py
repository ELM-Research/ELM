import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import MultipleLocator
from math import ceil

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
GRID_MAJOR = (0.85, 0.25, 0.25, 0.45)
GRID_MINOR = (0.90, 0.70, 0.70, 0.35)
SIGNAL     = (0.0, 0.0, 0.55)
BACKGROUND = (1.0, 0.98, 0.96)

class VizManager:
    def __init__(self, out_dir="debug_plots"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def has_nan(self, x: np.ndarray) -> bool:
        return np.isnan(x).any()

    def plot_signals(self, signals, file_name):
        names = list(signals.keys())
        n = len(names)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
        if n == 1:
            axes = np.array([axes])
        for i, name in enumerate(names):
            y = np.asarray(signals[name])
            x = np.arange(len(y))
            ax = axes[i]
            if self.has_nan(y):
                mask = ~np.isnan(y)
                ax.plot(x[mask], y[mask])
            else:
                ax.plot(x, y)
            ax.set_ylabel(name)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"{file_name}.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_train_val_loss(train_loss, val_loss=None, dir_path=None):
        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, "b", label="Training loss")
        if val_loss is not None:
            plt.plot(epochs, val_loss, "r", label="Validation loss")
            plt.title("Training and Validation Loss")
        else:
            plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{dir_path}/train_val_loss.png")
        plt.close()

    def _make_grid(self, ax, x_lim, y_lim):
        """Apply standard ECG grid: 0.2s major, 0.04s minor."""
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which='major', linewidth=0.4, color=GRID_MAJOR)
        ax.grid(which='minor', linewidth=0.2, color=GRID_MINOR)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def plot_ecg(self, ecg, sample_rate=250, lead_names=None, lead_order=None,
                columns=2, row_height=1.5, title=None, style='color'):
        """Create a clinical multi-lead ECG figure.

        Args:
            ecg:         (leads, samples) array.
            sample_rate: Hz.
            lead_names:  List of lead label strings.
            lead_order:  Index order for display.
            columns:     Number of display columns.
            row_height:  mV range per row (default 1.5 mV above/below baseline).
            title:       Optional title string.
            style:       'color' (default) or 'bw'.

        Returns:
            matplotlib Figure.
        """
        ecg = np.asarray(ecg)
        if ecg.ndim == 1:
            ecg = ecg[np.newaxis, :]

        n_leads, n_samples = ecg.shape
        lead_order = lead_order or list(range(n_leads))
        lead_names = lead_names or LEAD_NAMES[:n_leads]
        secs = n_samples / sample_rate
        rows = ceil(len(lead_order) / columns)
        t = np.linspace(0, secs, n_samples, endpoint=False)

        sig_color = (0, 0, 0) if style == 'bw' else SIGNAL
        bg_color = (1, 1, 1) if style == 'bw' else BACKGROUND

        fig_w = secs * columns * 2.0
        fig_h = rows * row_height * 1.1
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        x_max = secs * columns
        y_min = -rows * row_height
        y_max = row_height * 0.25

        self._make_grid(ax, (0, x_max), (y_min, y_max))

        for idx, lead_idx in enumerate(lead_order):
            col = idx // rows
            row = idx % rows
            x_off = col * secs
            y_off = -row * row_height

            # Signal
            ax.plot(t + x_off, ecg[lead_idx] + y_off,
                    linewidth=0.6, color=sig_color, zorder=3)

            # Lead label
            ax.text(x_off + 0.05, y_off + row_height * 0.3,
                    lead_names[lead_idx], fontsize=8, fontweight='bold',
                    color=(0.2, 0.2, 0.2), zorder=4)

            # Column separator
            if col > 0:
                ax.axvline(x_off, color=(0.5, 0.5, 0.5), linewidth=0.3,
                        linestyle='--', zorder=2)

        if title:
            fig.suptitle(title, fontsize=10, y=0.98)

        fig.tight_layout(pad=0.3)
        return fig

    def get_plot_as_image(self, ecg, dpi=150, **kwargs):
        """Render ECG plot to a numpy RGB array.

        Args:
            ecg:    (leads, samples) array.
            dpi:    Output resolution.
            **kwargs: Forwarded to plot_ecg().

        Returns:
            (H, W, 3) uint8 numpy array.
        """
        fig = self.plot_ecg(ecg, **kwargs)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def plot_2d_ecg(self, ecg, path, dpi=150, fmt=None, **kwargs):
        """Render and save ECG to file (png/svg/pdf auto-detected from extension).

        Args:
            ecg:    (leads, samples) array.
            path:   Output filepath (extension determines format).
            dpi:    Resolution for raster formats.
            **kwargs: Forwarded to plot_ecg().
        """
        fig = self.plot_ecg(ecg, **kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor(),
                    format=fmt)
        plt.close(fig)
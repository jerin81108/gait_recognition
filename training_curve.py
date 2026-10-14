"""
training_curve.py
-----------------
Plots training history (loss & accuracy) against constant expected output targets.

Expected output values are declared as MODULE-LEVEL CONSTANTS at the top of this file.
To adjust targets, change only the EXPECTED_* constants below — no other code needs to change.

Usage:
    python training_curve.py                     # uses embedded fallback log
    python training_curve.py --log train_log.txt # parses a plain-text log produced by train.py
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONSTANT EXPECTED OUTPUT TARGETS
# Adjust these values to change the horizontal "expected output" reference lines
# shown on the plots.  Nothing else in this file needs to be edited.
# ---------------------------------------------------------------------------

# Loss targets
EXPECTED_FINAL_VAL_LOSS    = 1.0      # Target val-loss the model should reach
EXPECTED_BASELINE_LOSS     = 2.08     # Untrained / random-chance baseline loss

# Accuracy targets (percentages)
EXPECTED_ID_ACCURACY       = 75.0     # Target person-identification accuracy (%)
EXPECTED_GENDER_ACCURACY   = 85.0     # Target gender-classification accuracy (%)
EXPECTED_RANDOM_ID_CHANCE  = 25.0     # Random-chance ID accuracy (4-class problem, %)
EXPECTED_RANDOM_GEN_CHANCE = 50.0     # Random-chance gender accuracy (binary, %)

# Output defaults
OUT_DIR  = "models"
OUT_FILE = "training_curves_with_targets.png"

# ---------------------------------------------------------------------------
# FALLBACK LOG (used when no external log file is passed)
# Replace this string with your actual training output, or pass --log <file>.
# ---------------------------------------------------------------------------
_FALLBACK_LOG = """
[INFO] Epoch 001 | T-Loss: 2.1865 | V-Loss: 2.0771 | ID-Acc: 25.0% | G-Acc: 62.5%
[INFO] Epoch 010 | T-Loss: 1.5646 | V-Loss: 1.6161 | ID-Acc: 62.5% | G-Acc: 75.0%
[INFO] Epoch 020 | T-Loss: 1.1861 | V-Loss: 1.3360 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 030 | T-Loss: 1.0784 | V-Loss: 1.2464 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 040 | T-Loss: 0.9913 | V-Loss: 1.2065 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 050 | T-Loss: 1.0154 | V-Loss: 1.2522 | ID-Acc: 62.5% | G-Acc: 75.0%
[INFO] Epoch 060 | T-Loss: 0.9549 | V-Loss: 1.3249 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 070 | T-Loss: 0.9792 | V-Loss: 1.1849 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 080 | T-Loss: 0.9330 | V-Loss: 1.3911 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 090 | T-Loss: 0.9588 | V-Loss: 1.2147 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 100 | T-Loss: 0.9177 | V-Loss: 1.1205 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 110 | T-Loss: 0.9511 | V-Loss: 1.1855 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 120 | T-Loss: 0.9113 | V-Loss: 1.1802 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 130 | T-Loss: 0.9522 | V-Loss: 1.3428 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 140 | T-Loss: 0.9330 | V-Loss: 1.1862 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 150 | T-Loss: 0.9127 | V-Loss: 1.1004 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 160 | T-Loss: 0.9357 | V-Loss: 1.1133 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 170 | T-Loss: 0.9292 | V-Loss: 1.1945 | ID-Acc: 62.5% | G-Acc: 87.5%
[INFO] Epoch 180 | T-Loss: 0.9055 | V-Loss: 1.1037 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 190 | T-Loss: 0.8702 | V-Loss: 1.1273 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 200 | T-Loss: 0.8831 | V-Loss: 1.0978 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Epoch 210 | T-Loss: 0.9247 | V-Loss: 1.0911 | ID-Acc: 75.0% | G-Acc: 87.5%
[INFO] Early Stopping at epoch 216 (Good Fit reached).
[INFO] COMPLETE. Best Val Loss: 1.0809
"""

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
_EPOCH_RE      = re.compile(
    r'Epoch\s+(\d+)\s+\|\s+T-Loss:\s+([\d.]+)\s+\|\s+V-Loss:\s+([\d.]+)'
    r'\s+\|\s+ID-Acc:\s+([\d.]+)%\s+\|\s+G-Acc:\s+([\d.]+)%'
)
_EARLY_STOP_RE = re.compile(r'Early Stopping at epoch\s+(\d+)')
_BEST_LOSS_RE  = re.compile(r'Best Val Loss:\s+([\d.]+)')


def parse_log(text: str):
    """Parse training log text; return arrays of checkpoint epochs and metrics."""
    epochs, t_loss, v_loss, id_acc, g_acc = [], [], [], [], []
    early_stop_epoch = None
    best_val_loss    = None

    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            t_loss.append(float(m.group(2)))
            v_loss.append(float(m.group(3)))
            id_acc.append(float(m.group(4)))
            g_acc.append(float(m.group(5)))
            continue

        m2 = _EARLY_STOP_RE.search(line)
        if m2:
            early_stop_epoch = int(m2.group(1))

        m3 = _BEST_LOSS_RE.search(line)
        if m3:
            best_val_loss = float(m3.group(1))

    return (
        np.array(epochs, dtype=float),
        np.array(t_loss),
        np.array(v_loss),
        np.array(id_acc),
        np.array(g_acc),
        early_stop_epoch,
        best_val_loss,
    )


def load_history(log_path: str = None):
    """Load training data from a log file, or fall back to the embedded string."""
    if log_path and os.path.isfile(log_path):
        with open(log_path, 'r') as f:
            text = f.read()
        print(f"[INFO] Reading log from: {log_path}")
    else:
        print("[INFO] No external log provided — using embedded fallback log.")
        text = _FALLBACK_LOG
    return parse_log(text)


def interpolate(checkpoints, values, total_epochs):
    """Linearly interpolate recorded checkpoint values to every epoch."""
    all_e = np.arange(1, total_epochs + 1)
    return all_e, np.interp(all_e, checkpoints, values)


# ---------------------------------------------------------------------------
# MAIN PLOT FUNCTION
# ---------------------------------------------------------------------------
def plot_training_curves(log_path=None, out_dir=OUT_DIR, out_file=OUT_FILE):
    epochs, t_loss, v_loss, id_acc, g_acc, early_stop, best_loss = load_history(log_path)

    if len(epochs) == 0:
        print("[ERROR] No epoch data found in log. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    final_epoch = early_stop if early_stop else int(epochs[-1])
    best_loss   = best_loss  if best_loss  is not None else float(v_loss[-1])

    # Ensure the final epoch is represented in the checkpoint arrays
    if epochs[-1] < final_epoch:
        epochs = np.append(epochs, final_epoch)
        t_loss = np.append(t_loss, t_loss[-1])
        v_loss = np.append(v_loss, best_loss)
        id_acc = np.append(id_acc, id_acc[-1])
        g_acc  = np.append(g_acc,  g_acc[-1])

    all_e, t_loss_i = interpolate(epochs, t_loss, final_epoch)
    _,     v_loss_i = interpolate(epochs, v_loss, final_epoch)
    _,     id_acc_i = interpolate(epochs, id_acc, final_epoch)
    _,     g_acc_i  = interpolate(epochs, g_acc,  final_epoch)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)

    # ── Subplot 1: Loss ────────────────────────────────────────────────────
    ax1.plot(all_e, t_loss_i, label='Train Loss (Actual)',
             color='#1f77b4', lw=2.2)
    ax1.plot(all_e, v_loss_i, label='Val Loss (Actual)',
             color='#ff7f0e', lw=2.2)

    # Constant expected output lines — driven by module-level constants
    ax1.axhline(y=EXPECTED_FINAL_VAL_LOSS,
                color='#2ca02c', linestyle='-', linewidth=2.0, alpha=0.9,
                label=f'Expected Val-Loss Target (constant = {EXPECTED_FINAL_VAL_LOSS:.2f})')
    ax1.axhline(y=EXPECTED_BASELINE_LOSS,
                color='#7f7f7f', linestyle='--', linewidth=1.2, alpha=0.7,
                label=f'Untrained Baseline (constant = {EXPECTED_BASELINE_LOSS:.2f})')

    ax1.scatter([final_epoch], [best_loss], color='red', s=70, zorder=5)
    ax1.annotate(
        f'Achieved: {best_loss:.4f}\n(Early Stop @ Epoch {final_epoch})',
        xy=(final_epoch, best_loss),
        xytext=(max(1, final_epoch - 95), 1.55),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d4b106"),
    )

    ax1.set_title('Loss Reduction vs Constant Expected Target',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Loss (CrossEntropy + Label Smoothing)', fontsize=10)
    ax1.set_ylim(0.5, 2.5)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ── Subplot 2: Accuracy ────────────────────────────────────────────────
    ax2.plot(all_e, id_acc_i,
             label=f'ID Accuracy — Actual (final: {id_acc[-1]:.1f}%)',
             color='#1f77b4', lw=2.2)
    ax2.plot(all_e, g_acc_i,
             label=f'Gender Accuracy — Actual (final: {g_acc[-1]:.1f}%)',
             color='#ff7f0e', lw=2.2)

    # Constant expected output lines — driven by module-level constants
    ax2.axhline(y=EXPECTED_ID_ACCURACY,
                color='#1f77b4', linestyle='-', linewidth=2.0, alpha=0.85,
                label=f'Expected ID Target (constant = {EXPECTED_ID_ACCURACY:.1f}%)')
    ax2.axhline(y=EXPECTED_GENDER_ACCURACY,
                color='#ff7f0e', linestyle='-', linewidth=2.0, alpha=0.85,
                label=f'Expected Gender Target (constant = {EXPECTED_GENDER_ACCURACY:.1f}%)')

    # Random-chance baselines — also constants
    ax2.axhline(y=EXPECTED_RANDOM_ID_CHANCE,
                color='#d62728', linestyle=':', linewidth=1.3, alpha=0.7,
                label=f'Random ID Chance (constant = {EXPECTED_RANDOM_ID_CHANCE:.1f}%)')
    ax2.axhline(y=EXPECTED_RANDOM_GEN_CHANCE,
                color='#8c564b', linestyle=':', linewidth=1.3, alpha=0.7,
                label=f'Random Gender Chance (constant = {EXPECTED_RANDOM_GEN_CHANCE:.1f}%)')

    ax2.scatter([final_epoch], [id_acc[-1]], color='#1f77b4', s=60, zorder=5)
    ax2.scatter([final_epoch], [g_acc[-1]],  color='#ff7f0e', s=60, zorder=5)
    ax2.annotate(
        f'Achieved: {id_acc[-1]:.1f}%\n'
        f'(Target {"Reached" if id_acc[-1] >= EXPECTED_ID_ACCURACY else "Not Reached"})',
        xy=(final_epoch, id_acc[-1]),
        xytext=(max(1, final_epoch - 75), id_acc[-1] - 9),
        arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1, headwidth=5),
        fontsize=8.5, fontweight='bold',
    )
    ax2.annotate(
        f'Achieved: {g_acc[-1]:.1f}%\n'
        f'(Target {"Exceeded" if g_acc[-1] > EXPECTED_GENDER_ACCURACY else "Reached"})',
        xy=(final_epoch, g_acc[-1]),
        xytext=(max(1, final_epoch - 75), g_acc[-1] + 4),
        arrowprops=dict(facecolor='#ff7f0e', shrink=0.05, width=1, headwidth=5),
        fontsize=8.5, fontweight='bold',
    )

    ax2.set_title('Validation Accuracy vs Constant Expected Targets',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_ylim(15, 105)
    ax2.legend(loc='lower right', fontsize=8.2, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_file)
    plt.savefig(save_path)
    print(f"[INFO] Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves with constant expected output targets."
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to a plain-text training log produced by train.py. "
             "If omitted, the embedded fallback log is used.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=OUT_DIR,
        help=f"Directory to save the output PNG (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--out-file", type=str, default=OUT_FILE,
        help=f"Output filename (default: {OUT_FILE}).",
    )
    args = parser.parse_args()
    plot_training_curves(log_path=args.log, out_dir=args.out_dir, out_file=args.out_file)

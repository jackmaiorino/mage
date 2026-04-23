"""Plot value head accuracy and separation over time for Pauper-Standard-Wide."""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
from datetime import datetime

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Mage.Server.Plugins", "Mage.Player.AIRL", "src", "mage", "player", "ai", "rl",
    "profiles", "Pauper-Standard-Wide", "logs", "stats", "value_accuracy.csv",
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "value_head_trajectory.png")

df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").dt.tz_convert("US/Eastern")
df["separation"] = df["avg_win_value"] - df["avg_loss_value"]

# Smooth with rolling mean to cut noise
WINDOW = 30
df["val_acc_smooth"] = df["value_accuracy"].rolling(WINDOW, min_periods=1).mean()
df["sep_smooth"] = df["separation"].rolling(WINDOW, min_periods=1).mean()

# Key event markers (approx EDT):
COLLAPSE_END = pd.Timestamp("2026-04-21 23:12", tz="US/Eastern")          # val_acc ~0.07
PPO_FIX = pd.Timestamp("2026-04-22 01:15", tz="US/Eastern")                # ratio clamp fix applied
UNSTICK_ATTEMPT = pd.Timestamp("2026-04-22 11:18", tz="US/Eastern")        # targeted heal
FRESH_START = pd.Timestamp("2026-04-22 11:41", tz="US/Eastern")            # restart from scratch

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Top: value accuracy
ax1 = axes[0]
ax1.plot(df["timestamp"], df["value_accuracy"], color="lightblue", linewidth=0.5, label="raw")
ax1.plot(df["timestamp"], df["val_acc_smooth"], color="navy", linewidth=2.0, label=f"rolling mean (n={WINDOW})")
ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="random baseline (0.5)")
ax1.axhline(0.75, color="green", linestyle=":", alpha=0.5, label="healthy target (0.75)")
ax1.set_ylabel("value_accuracy (sign-correct fraction)")
ax1.set_title("Pauper-Standard-Wide value head recovery")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(alpha=0.3)

# Bottom: separation (win_value - loss_value)
ax2 = axes[1]
ax2.plot(df["timestamp"], df["separation"], color="mistyrose", linewidth=0.5, label="raw")
ax2.plot(df["timestamp"], df["sep_smooth"], color="darkred", linewidth=2.0, label=f"rolling mean (n={WINDOW})")
ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax2.axhline(0.5, color="green", linestyle=":", alpha=0.6, label="MCTS-ready threshold (0.5)")
ax2.set_ylabel("avg_win_value - avg_loss_value")
ax2.set_xlabel("Time (EDT)")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_ylim(-0.6, 1.2)
ax2.grid(alpha=0.3)

# Event markers on both axes
for ax in axes:
    for t, lbl, color in [
        (COLLAPSE_END, "value collapse discovered", "red"),
        (PPO_FIX, "PPO ratio clamp fix", "orange"),
        (UNSTICK_ATTEMPT, "unstick attempt", "purple"),
        (FRESH_START, "fresh start", "green"),
    ]:
        ax.axvline(t, color=color, linestyle="-", alpha=0.7, linewidth=1.5)
        ax.annotate(lbl, xy=(t, ax.get_ylim()[1] * 0.95), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=color,
                    rotation=90, va="top")

# Clip x-axis to where we have data
data_start = df["timestamp"].quantile(0.01)
ax1.set_xlim(data_start, df["timestamp"].max())

plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Rows plotted: {len(df)}")
print(f"Latest val_acc: {df['value_accuracy'].iloc[-1]:.3f}  separation: {df['separation'].iloc[-1]:.3f}")

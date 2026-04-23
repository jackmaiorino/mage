"""Close-up: fresh-start value head trajectory only."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Mage.Server.Plugins", "Mage.Player.AIRL", "src", "mage", "player", "ai", "rl",
    "profiles", "Pauper-Standard-Wide", "logs", "stats", "value_accuracy.csv",
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "value_head_fresh_zoom.png")

df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").dt.tz_convert("US/Eastern")
df["separation"] = df["avg_win_value"] - df["avg_loss_value"]

FRESH_START = pd.Timestamp("2026-04-22 11:41", tz="US/Eastern")
fresh = df[df["timestamp"] >= FRESH_START].copy()
fresh["minutes"] = (fresh["timestamp"] - FRESH_START).dt.total_seconds() / 60.0

# Smooth lightly
fresh["val_acc_smooth"] = fresh["value_accuracy"].rolling(20, min_periods=1).mean()
fresh["sep_smooth"] = fresh["separation"].rolling(20, min_periods=1).mean()

# Also compute rolling slope over 15-min windows
fresh["val_acc_slope"] = fresh["val_acc_smooth"].diff(periods=60)  # ~change over last 60 samples
fresh["sep_slope"] = fresh["sep_smooth"].diff(periods=60)

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# Panel 1: val_acc
ax = axes[0]
ax.plot(fresh["minutes"], fresh["value_accuracy"], color="lightblue", linewidth=0.4)
ax.plot(fresh["minutes"], fresh["val_acc_smooth"], color="navy", linewidth=2.0, label="val_acc (rolling)")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="random baseline")
ax.axhline(0.75, color="green", linestyle=":", alpha=0.5, label="healthy target")
ax.set_ylabel("value_accuracy")
ax.set_title("Pauper-Standard-Wide — fresh-start zoom")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 2: separation
ax = axes[1]
ax.plot(fresh["minutes"], fresh["separation"], color="mistyrose", linewidth=0.4)
ax.plot(fresh["minutes"], fresh["sep_smooth"], color="darkred", linewidth=2.0, label="separation (rolling)")
ax.axhline(0.5, color="green", linestyle=":", alpha=0.5, label="MCTS-ready threshold")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("win_value - loss_value")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: slopes (rate of change)
ax = axes[2]
ax.plot(fresh["minutes"], fresh["val_acc_slope"], color="navy", linewidth=1.5, label="val_acc Δ over ~60 samples")
ax.plot(fresh["minutes"], fresh["sep_slope"], color="darkred", linewidth=1.5, label="separation Δ over ~60 samples")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_ylabel("rolling change (slope)")
ax.set_xlabel("Minutes since fresh-start")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Rows since fresh start: {len(fresh)}")
print(f"val_acc first 5min: {fresh.head(20)['value_accuracy'].mean():.3f}")
print(f"val_acc last 5min: {fresh.tail(20)['value_accuracy'].mean():.3f}")
print(f"separation first 5min: {fresh.head(20)['separation'].mean():.3f}")
print(f"separation last 5min: {fresh.tail(20)['separation'].mean():.3f}")

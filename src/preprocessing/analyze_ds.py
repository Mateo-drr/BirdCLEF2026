
"""
BirdCLEF 2026 – Pantanal EDA  (torchaudio version)
Run from competition root or a Kaggle notebook.
"""

import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parents[2] / "dataset"
TRAIN_CSV   = BASE / "train.csv"
TAXONOMY    = BASE / "taxonomy.csv"
SAMPLE_SUB  = BASE / "sample_submission.csv"
SS_LABELS   = BASE / "train_soundscapes_labels.csv"
TRAIN_AUDIO = BASE / "train_audio"
TRAIN_SC    = BASE / "train_soundscapes"

TARGET_SR   = 32_000
N_FFT       = 1024
HOP         = 256
N_MELS      = 128
AUDIO_N     = 4       # train_audio clips to sample
SC_N        = 2       # soundscapes to sample
SC_DURATION = 15      # seconds of soundscape to plot

# ── 1. CSVs ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("CSV OVERVIEW")
print("=" * 70)

def show_df(name, df):
    print(f"\n── {name}  shape={df.shape}")
    #print(df.dtypes.to_string())
    #print(df.head(3).to_string())
    print(df.iloc[0])

train    = pd.read_csv(TRAIN_CSV)
taxonomy = pd.read_csv(TAXONOMY)
sub      = pd.read_csv(SAMPLE_SUB)

show_df("train.csv",         train)
show_df("taxonomy.csv",      taxonomy)
#show_df("sample_submission", sub)

ssl = None
if SS_LABELS.exists():
    ssl = pd.read_csv(SS_LABELS)
    show_df("train_soundscapes_labels.csv", ssl)
else:
    print("\ntrain_soundscapes_labels.csv not found – skipping")
    
unique_anim_ids_train = set(train["primary_label"])
unique_anim_ids_ssl = []
for labels in ssl["primary_label"]:
    result = [x.strip() for x in labels.split(';')]
    unique_anim_ids_ssl += result
unique_anim_ids_ssl = set(unique_anim_ids_ssl)
    
# one file can have many labels for the same window
clean = {}
for _, row in ssl.iterrows():
    key = (row["filename"], row["start"], row["end"])
    # get the labels in that row
    labels = [l.strip() for l in row["primary_label"].split(";")]
    if key not in clean:
        clean[key] = set()
    clean[key].update(labels) # add any missing labels to the file window set
# convert to dataframe
rows = []
for (filename, start, end), labels in clean.items():
    rows.append({
        "filename": filename,
        "start": start,
        "end": end,
        "label_list": sorted(labels)
    })

sc_clean = pd.DataFrame(rows)
print(sc_clean.shape)
print(sc_clean.iloc[0])

all_labels = []
for lbls in sc_clean["label_list"]:
    all_labels += lbls

counts = Counter(all_labels)
labels, freqs = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))
plt.figure(figsize=(14, 5))
plt.bar(labels, freqs, edgecolor='black')
plt.xticks(rotation=90)
plt.xlabel("Label")
plt.ylabel("Number of windows")
plt.title("Label Frequencies")
plt.tight_layout()
plt.show()


# flatten sc_clean labels into one row per label
sc_rows = []
for _, row in sc_clean.iterrows():
    for lbl in row["label_list"]:
        sc_rows.append({"primary_label": lbl, "source": "soundscape"})

# train.csv already has one label per row
train_rows = [{"primary_label": lbl, "source": "train_audio"}
              for lbl in train["primary_label"]]

combined = pd.DataFrame(sc_rows + train_rows)
print(combined.shape)
print(combined["source"].value_counts())

# count per label per source
counts = combined.groupby(["primary_label", "source"]).size().unstack(fill_value=0)
counts["total"] = counts.sum(axis=1)
counts = counts.sort_values("total", ascending=False)

# plot stacked bar
fig, ax = plt.subplots(figsize=(18, 5))
counts[["train_audio", "soundscape"]].plot(kind="bar", stacked=True, ax=ax, edgecolor="black")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
ax.set_title("Label Frequencies — train_audio vs soundscape")
ax.tick_params(axis="x", rotation=90, labelsize=7)
plt.tight_layout()
plt.show()

# summary table
print("\nTop 10 by total count:")
print(counts.head(10))
print("\nLabels only in train_audio (not in soundscapes):")
only_train = counts[counts["soundscape"] == 0].index.tolist()
print(len(only_train), only_train[:10])
print("\nLabels only in soundscapes (not in train_audio):")
only_sc = counts[counts["train_audio"] == 0].index.tolist()
print(len(only_sc), only_sc[:10])


# ── 2. train.csv summaries ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAIN.CSV STATS")
print("=" * 70)
print(f"Total recordings      : {len(train):,}")
print(f"Unique primary_label  : {train['primary_label'].nunique()}")
print(f"Collections           : {train['collection'].value_counts().to_dict()}")
print(f"\nRating distribution:\n{train['rating'].value_counts().sort_index()}")
print(f"\nTop 10 species by count:\n{train['primary_label'].value_counts().head(10)}")
print(f"\nRecordings with lat/lon: {train[['latitude','longitude']].notna().all(axis=1).sum():,}")
print(f"\nTaxonomy class counts:\n{taxonomy['class_name'].value_counts()}")

if ssl is not None:
    print("\n── Soundscape label stats")
    print(f"Labeled segments      : {len(ssl):,}")
    print(f"Unique soundscapes    : {ssl['filename'].nunique()}")
    sc_species = ssl["primary_label"].str.split(";").explode().str.strip()
    print(f"Unique species in ssl : {sc_species.nunique()}")
    print(f"Top 10:\n{sc_species.value_counts().head(10)}")

# ── 3. Audio helpers ───────────────────────────────────────────────────────────
def load_audio(path, max_seconds=None):
    """Load ogg, resample to TARGET_SR, mix to mono. Returns 1-D tensor."""
    wav, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    wav = wav.mean(dim=0)                           # mono
    if max_seconds is not None:
        wav = wav[:int(max_seconds * TARGET_SR)]
    return wav, TARGET_SR


def acoustic_stats(wav, sr):
    """Compute spectral stats from a 1-D float tensor."""
    y   = wav.numpy().astype(np.float32)
    dur = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))

    # FFT-based stats
    fft      = np.abs(np.fft.rfft(y))
    freqs    = np.fft.rfftfreq(len(y), 1.0 / sr)
    dom_freq = float(freqs[np.argmax(fft)])

    # power-weighted centroid and bandwidth
    power    = fft ** 2
    total    = power.sum() + 1e-9
    centroid = float((freqs * power).sum() / total)
    bw       = float(np.sqrt((power * (freqs - centroid) ** 2).sum() / total))

    # zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)

    return dict(
        duration_s            = round(dur, 2),
        rms                   = round(rms, 5),
        dominant_freq_hz      = round(dom_freq, 1),
        spectral_centroid_hz  = round(centroid, 1),
        spectral_bandwidth_hz = round(bw, 1),
        zcr                   = round(zcr, 5),
    )


def make_specs(wav, sr):
    """Return (linear_db, mel_db, freq_axis_hz) as numpy arrays."""
    # linear spectrogram
    lin_transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=2.0)
    lin  = T.AmplitudeToDB()(lin_transform(wav)).numpy()
    freqs = np.linspace(0, sr / 2, lin.shape[0])

    # mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=50, f_max=sr // 2,
    )
    mel = T.AmplitudeToDB()(mel_transform(wav)).numpy()

    return lin, mel, freqs


def plot_clip(axes, wav, sr, title):
    ax_wave, ax_lin, ax_mel = axes
    y = wav.numpy()
    t = np.linspace(0, len(y) / sr, len(y))

    # waveform
    ax_wave.plot(t, y, linewidth=0.4, color="#4a9eff")
    ax_wave.set_title(title, fontsize=8)
    ax_wave.set_xlabel("time (s)")
    ax_wave.set_ylabel("amplitude")

    lin, mel, freq_axis = make_specs(wav, sr)
    times_lin = np.linspace(0, len(y) / sr, lin.shape[1])
    times_mel = np.linspace(0, len(y) / sr, mel.shape[1])

    # linear spectrogram
    ax_lin.pcolormesh(times_lin, freq_axis, lin, shading="auto", cmap="magma")
    ax_lin.set_ylabel("freq (Hz)")
    ax_lin.set_xlabel("time (s)")
    ax_lin.set_title("Spectrogram (linear Hz)", fontsize=7)

    # mel spectrogram
    ax_mel.pcolormesh(times_mel, np.arange(N_MELS), mel, shading="auto", cmap="magma")
    ax_mel.set_ylabel("mel bin")
    ax_mel.set_xlabel("time (s)")
    ax_mel.set_title("Mel spectrogram", fontsize=7)


# ── 4. Sample train_audio ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAIN_AUDIO CLIPS")
print("=" * 70)

audio_files  = list(TRAIN_AUDIO.rglob("*.ogg"))
random.seed(42)
sample_audio = random.sample(audio_files, min(AUDIO_N, len(audio_files)))

rows = []
fig, axes = plt.subplots(AUDIO_N, 3, figsize=(15, 3 * AUDIO_N))
if AUDIO_N == 1:
    axes = [axes]

for i, fpath in enumerate(sample_audio):
    fname = fpath.name
    match = train[train["filename"] == fname]
    label = match["primary_label"].values[0] if len(match) else "unknown"

    wav, sr = load_audio(fpath)
    stats   = acoustic_stats(wav, sr)
    stats.update({"file": fname, "label": label})
    rows.append(stats)

    print(f"\n{fname}  label={label}")
    for k, v in stats.items():
        if k not in ("file", "label"):
            print(f"  {k:<28} {v}")

    plot_clip(axes[i], wav, sr, f"{label} | {fname}")

plt.tight_layout()
plt.show()

# ── 5. Sample train_soundscapes ────────────────────────────────────────────────
if TRAIN_SC.exists():
    print("\n" + "=" * 70)
    print(f"TRAIN_SOUNDSCAPES (stats on full clip, plot = first {SC_DURATION}s)")
    print("=" * 70)

    sc_files  = list(TRAIN_SC.rglob("*.ogg"))
    sample_sc = random.sample(sc_files, min(SC_N, len(sc_files)))

    fig2, axes2 = plt.subplots(SC_N, 3, figsize=(15, 3 * SC_N))
    if SC_N == 1:
        axes2 = [axes2]

    for i, fpath in enumerate(sample_sc):
        wav_full, sr = load_audio(fpath)
        stats = acoustic_stats(wav_full, sr)
        print(f"\n{fpath.name}")
        for k, v in stats.items():
            print(f"  {k:<28} {v}")

        wav_plot = wav_full[:int(SC_DURATION * sr)]
        plot_clip(axes2[i], wav_plot, sr, f"{fpath.name} (first {SC_DURATION}s)")

    plt.tight_layout()
    plt.show()
else:
    print("\ntrain_soundscapes/ not found – skipping")

# ── 6. Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AUDIO STATS SUMMARY (train_audio samples)")
print("=" * 70)
cols = ["label", "duration_s", "spectral_centroid_hz",
        "spectral_bandwidth_hz", "dominant_freq_hz", "zcr"]
print(pd.DataFrame(rows)[cols].to_string(index=False))

print("\nDone.")
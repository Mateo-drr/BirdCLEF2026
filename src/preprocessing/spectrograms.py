import numpy as np
import scipy.signal as signal
import librosa
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

SR = 32000
HOP_LENGTH = 320    # 10ms
WIN_LENGTH = 640    # 20ms
N_FFT = 1024
N_MELS = 128

def process_audio_files(files: list[Path],
                        encoder,
                        save_dir: Path,
                        split: str,
                        sr: int = SR):

    # mel filterbank (HTK formula)
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=sr // 2,
        htk=True,       # HTK formula
        norm=None,      # no filterbank normalization
    )  # shape: (n_mels, n_fft // 2 + 1)

    hann = signal.windows.hann(WIN_LENGTH, sym=False)  # periodic hann
    window_scale = 1.0 / hann.sum()                    # scipy STFT convention

    for i, file in enumerate(tqdm(files, desc="Processing files")):

        data, og_sr = torchaudio.load(file[0])

        if data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)

        if og_sr != sr:
            resampler = T.Resample(orig_freq=og_sr, new_freq=sr)
            data = resampler(data)

        audio = data.squeeze().numpy()  # (samples,)

        # STFT — scipy convention, uncentered, hann window
        _, _, stft = signal.stft(
            audio,
            fs=sr,
            window=hann,
            nperseg=WIN_LENGTH,
            noverlap=WIN_LENGTH - HOP_LENGTH,
            nfft=N_FFT,
            boundary=None,      # uncentered: no padding at boundaries
            padded=False,
        )
        # stft shape: (n_fft//2 + 1, n_frames), already scaled by 1/sum(window)

        mag = np.abs(stft)                          # magnitude spectrogram
        mel = mel_fb @ mag                          # (n_mels, n_frames)
        log_mel = np.log(np.maximum(mel, 1e-5)) * 0.1

        lbl = encoder.transform(list(file[1]))
        result = (log_mel, lbl)
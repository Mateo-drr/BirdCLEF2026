
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

def process_audio_files(files: list[Path],
                        encoder,
                        save_dir: Path,
                        split: str,
                        sr: int=44100):
    """
    Calculate the mel spectrograms for all the audio files in "files" list,
    and save them in the specified "save_dir" / "split"

    Args:
        files (list[Path]): list of audio files to process
        encoder (LabelEncoder): encoder to convert the str labels to int
        save_dir (Path): directory to save the processed files
        split (str): subdirectory to specify the split where to save
        sr (int, optional): Sampling rate of the audio files. Default 44100 Hz

    """

    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,  # 44100
        n_fft=2048,  # FFT window length
        hop_length=1024,  # 50% of n_fft (2048/2 = 1024)
        f_min=0,  # Minimum frequency
        f_max=sr // 2 if sr != 44100 else 16000,  # Maximum frequency (16 kHz)
        n_mels=128,  # Number of mel frequency bands
    )

    to_db = torchaudio.transforms.AmplitudeToDB()

    # TODO we can try this even simpler method
    # mfcc_transform = torchaudio.transforms.MFCC(
    #     sample_rate=sr,
    #     n_mfcc=13,
    #     melkwargs={
    #         "n_fft": 2048,
    #         "hop_length": 1024,
    #         "f_min": 0,
    #         "f_max": sr//2 if sr != 44100 else 16000,
    #         "n_mels": 128
    #     })

    for i, file in enumerate(tqdm(files, desc="Processing files")):

        data, og_sr = torchaudio.load(file[0])

        if data.shape[0] > 1:  # turn into mono
            data = data.mean(dim=0, keepdim=True)

        if og_sr != sr:
            resampler = T.Resample(orig_freq=og_sr, new_freq=sr)
            data = resampler(data)

        lbl = encoder.transform(list(file[1]))

        result = (to_db(spec(data)), lbl)

        # save_data(result, save_dir, split, f"spec_{i:05d}")


if __name__ == "__main__":

    audio_files = []



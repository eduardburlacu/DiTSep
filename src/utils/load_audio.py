import torch
import librosa

def load_audio(wav_path):
    wav = librosa.load(wav_path, sr=16000)[0]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)
    return wav
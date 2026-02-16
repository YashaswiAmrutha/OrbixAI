import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

model = WhisperModel("base")

def listen_and_transcribe(duration=5):
    print("🎤 Listening...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wav.write("input.wav", fs, audio)
    segments, info = model.transcribe("input.wav")
    result_text = "".join([segment.text for segment in segments]).strip()

    return result_text


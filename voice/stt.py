import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

model = whisper.load_model("base")

def listen_and_transcribe(duration=5):
    print("🎤 Listening...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wav.write("input.wav", fs, audio)
    result = model.transcribe("input.wav")

    return result["text"].strip()


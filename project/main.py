import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from numpy.fft import fftshift, ifft
from scipy.io import wavfile
from scipy.signal import fftconvolve, stft, windows


import config.parameters
import processing.callback




# 主程序：开启音频流
def main():
    global fs_clean

    # 指定录音设备
    sd.default.device = 'MCHStreamer Lite I2S Spdif'

    # 启动音频流
    with sd.Stream(callback=audio_callback, channels=(10,10), samplerate=fs_clean, blocksize=blockLenSmp):
        print("Processing audio stream... Press Ctrl+C to stop.")
        sd.sleep(int(60 * 1000))  # 运行60秒

if __name__ == "__main__":
    main()

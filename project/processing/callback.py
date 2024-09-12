import numpy as np
import sounddevice as sd
from numpy.fft import ifft, fftshift
from scipy.signal import windows, stft, fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt


from .filters.fdMWF import fdMWF
from config.parameters import *




# 实时音频处理回调函数
def audio_callback(indata, outdata, frames, time, status):
    global clean_signal, fs_clean, blockLenSmp, hopSizeSmp, filterLenSmp, regulWeight, win

    if status:
        print(status)

    # 选择麦克风通道数
    mic_channel_use = [0,1,2,3,4,5,6]
    clean_channel_use = 7
    # 获取麦克风的当前音频块

    mic_signal = indata[:, mic_channel_use]
    clean_signal = indata[:, clean_channel_use]


    # 调整参考信号长度与当前麦克风信号长度一致
    if len(clean_signal) < len(mic_signal):
        mic_signal = mic_signal[:len(clean_signal)]
    else:
        clean_signal = clean_signal[:len(mic_signal)]

    # 调用 fdMWF 函数计算滤波器系数
    h, irShort, h_full = fdMWF(clean_signal, mic_signal, fs_clean, blockLenSmp, hopSizeSmp, filterLenSmp, regulWeight, win)





  #####################
#卷积这个信号，输出到9-10通道

    # output_left = np.zeros(indata.shape[0] + len(irShort[:,0]) - 1)
    # output_right = np.zeros(indata.shape[0] + len(irShort[:,1]) - 1)

    output_left = fftconvolve(indata[:, 7], irShort[:,0], mode='same')
    output_right = fftconvolve(indata[:, 7], irShort[:,1], mode='same')

    
    outdata[:, 8] = output_left
    outdata[:, 9] = output_right

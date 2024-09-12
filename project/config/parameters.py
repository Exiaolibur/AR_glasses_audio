import numpy as np
from scipy.signal import windows







# 参数设置
fs_clean = 44100
blockLenSmp = int(0.5 * fs_clean)
hopSizeSmp = 2048
filterLenSmp = int(blockLenSmp / 2)
regulWeight = 1e-4
win = np.sqrt(windows.hann(int(blockLenSmp), sym=False))

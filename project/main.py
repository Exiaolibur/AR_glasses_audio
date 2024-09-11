import dependencies


import config.parameters
import processing.callback




# 主程序：开启音频流
def main():
    global fs_clean

    # 指定录音设备
    sd.default.device = 'MCHStreamer Lite I2S Toslink'

    # 启动音频流
    with sd.Stream(callback=audio_callback, channels=(8,2), samplerate=fs_clean, blocksize=blockLenSmp):
        print("Processing audio stream... Press Ctrl+C to stop.")
        sd.sleep(int(60 * 1000))  # 运行60秒

if __name__ == "__main__":
    main()
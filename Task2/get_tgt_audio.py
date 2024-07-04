import pyaudio
import wave

# 设置参数
chunk = 1024
format = pyaudio.paInt16
channels = 2    ########## 然而tgt_audio是嘴边的双通道信号（双耳）
rate = 44100
record_seconds = 5
output_filename = "tgt_audio.wav"

# 创建PyAudio对象
audio = pyaudio.PyAudio()

# 列出所有音频输入设备
def list_input_devices():
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print(f"设备ID {i}: {dev['name']} - {dev['maxInputChannels']} channels")
# 检查和选择麦克风
list_input_devices()
device_index = int(input("请选择麦克风设备ID："))

# 打开音频流
try:
    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)
    print("开始录制声音...")

    frames = []  # 存储录制的音频数据

    # 录制声音
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("录制完成！")

    # 停止录制并关闭音频流
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存录制的音频数据为WAV文件
    wave_file = wave.open(output_filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    print("音频文件保存成功，文件名为：" + output_filename)
except Exception as e:
    print(f"录音过程中发生错误: {e}")

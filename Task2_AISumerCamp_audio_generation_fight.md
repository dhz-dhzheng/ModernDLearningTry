# Datawhale AI 夏令营 — 全球AI攻防挑战赛（AIGC 技术 — 语音方向）

> 目的：为了尝试其他语音生成模型，尝试将低电平、带噪的 WAV 音频处理为适合自动语音识别（ASR）模型的清晰音频，并提供多个可运行的示例脚本（Whisper / Vosk / SpeechBrain）与注释。

---

# 目录

1. 任务概述与推荐流程
2. 从参考音频到文字（ASR）

   * 2.1 工具速览
   * 2.2 OpenAI Whisper（安装 + 示例）
   * 2.3 Vosk（轻量离线 ASR，安装 + 示例）
3. 音频增强（噪声抑制、增益、带通、归一化）

   * 3.1 快速增益 + 降噪（pydub + noisereduce）
   * 3.2 一致响度（librosa）
   * 3.3 AI 降噪（SpeechBrain）
   * 3.4 带通滤波（保留语音频段）
4. 全流程示例脚本（增强 -> 转写）
5. 调试提示与常见问题
6. 参考与资源链接

---

# 1. 任务概述与推荐流程

目标：把低电平、噪声多的 WAV 文件转换为便于 ASR 识别的高质量音频。推荐的标准流程如下（按顺序执行）：

1. **格式与采样率标准化**：转换为单声道（mono）与 16 kHz（多数模型更稳定）。
2. **降噪**：先进行噪声估计与去噪（可选 AI 模型）。
3. **带通滤波**：保留语音频段（常用 300–3400 Hz）。
4. **增益与归一化**：避免削波（clipping），保持峰值或 RMS 在安全范围。
5. **转写（ASR）**：将处理后的音频输入模型（Whisper / Vosk / 等）。

注：处理流程可根据音频质量与模型要求做轻微调整（例如使用 48 kHz 高采样率的模型时，采样率应匹配模型）。

---

# 2. 从参考音频到文字（ASR）

## 2.1 工具速览（简表）

| 工具                    |             优点 | 缺点               |
| --------------------- | -------------: | ---------------- |
| Whisper               |  高准确率、多语种、用法简单 | 资源消耗较高（尤其 large） |
| Vosk                  |      轻量、离线、低资源 | 准确率通常低于 Whisper  |
| Coqui STT             |        支持自训练模型 | 开发活跃度与资源需评估      |
| SpeechBrain           |  丰富的预训练模型，研究友好 | 配置较复杂            |
| torchaudio pretrained | 与 PyTorch 集成良好 | 需要额外开发工作         |

---

## 2.2 OpenAI Whisper — 安装与示例

### 安装（在 Windows 或 WSL）

```bash
pip install -U openai-whisper
pip install torch
# ffmpeg 必需（用于音频格式转换/解码）
# Windows: 从 gyan.dev/ffmpeg/builds 下载并将 bin 目录加入系统 PATH
# 检查 ffmpeg 是否可用：
ffmpeg -version
```

### 简单转写示例（保存为 `wav_to_text_whisper.py`）

```python
import whisper

# 加载模型（可选: "tiny", "base", "small", "medium", "large"）
model = whisper.load_model("base")  # 权衡速度与准确率

# 待转写的 WAV 文件路径
audio_path = "sample.wav"

# 执行转写
result = model.transcribe(audio_path)

# 输出识别的文本
print("Recognized text:")
print(result["text"])
```

### 运行

```bash
python wav_to_text_whisper.py
```

### 说明与参数建议

* 速度优先：选择 `tiny` 或 `base`。
* 精度优先：选择 `large`（需较多显存/时间）。
* 语言指定：若需要强制指定语言，可传 `language="ja"` 等参数。
* Whisper 自动处理多种格式（WAV, MP3, M4A, FLAC 等），但建议先统一为 mono + 16 kHz。

---

## 2.3 Vosk — 轻量离线示例（适用于资源受限场景）

### 安装

```bash
pip install vosk soundfile numpy librosa
```

### 模型下载（示例：英语小模型）

```bash
mkdir models
# Linux/macOS 示例：
curl -L -o models/vosk-model-small-en-us-0.15.zip \
  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip models/vosk-model-small-en-us-0.15.zip -d models
```

> Windows 环境可手动在浏览器中下载并解压，或使用 PowerShell 的 `Invoke-WebRequest`。

### 示例脚本（保存为 `wav_to_text_vosk.py`）

```python
from vosk import Model, KaldiRecognizer
import soundfile as sf
import numpy as np
import json
import librosa

# 路径配置
audio_path = "sample.wav"
model_path = "models/vosk-model-small-en-us-0.15"

# 加载模型（只需加载一次）
model = Model(model_path)

# 读取音频（保持原始采样率）
data, sr = sf.read(audio_path)

# 若为立体声，转为单声道（取平均）
if data.ndim > 1:
    data = np.mean(data, axis=1)

# Vosk 常用要求：16000 Hz 单声道。必要时重采样
if sr != 16000:
    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    sr = 16000

# 将浮点数音频转换为 16-bit PCM（Vosk 接受原始 PCM bytes）
pcm16 = (data * 32767).astype(np.int16)

rec = KaldiRecognizer(model, sr)
rec.AcceptWaveform(pcm16.tobytes())
result = json.loads(rec.Result())
print(result.get("text", ""))
```

---

# 3. 音频增强（噪声抑制、增益、带通、归一化）

**关键目标**：增加语音信号能量（增益）、去除背景噪声、保留语音频段、避免削波。

## 3.1 快速增益 + 降噪（pydub + noisereduce）

### 安装

```bash
pip install pydub noisereduce soundfile numpy
# ffmpeg 仍需可用（供 pydub 使用）
```

### 示例脚本（保存为 `enhance_pydub_noisereduce.py`）

```python
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
import numpy as np
import os

input_path = "input.wav"
temp_path = "temp_resampled.wav"
output_path = "enhanced.wav"

# 1) 使用 pydub 做基础处理：转为单声道、24k/16k 重采样、加增益
audio = AudioSegment.from_file(input_path)
# 转单声道并设采样率为 16k
audio = audio.set_channels(1).set_frame_rate(16000)
# 增益示例：+10 dB
audio = audio + 10
# 导出临时 WAV 以供数值化处理
audio.export(temp_path, format="wav")

# 2) 使用 noisereduce 进行去噪（基于频谱估计）
data, rate = sf.read(temp_path)
if data.ndim > 1:
    data = np.mean(data, axis=1)
reduced = nr.reduce_noise(y=data, sr=rate)

# 3) 保存结果
sf.write(output_path, reduced, rate)
print("Enhanced saved to:", output_path)
```

### 注释

* pydub 的 `+` 操作是以 dB 为单位的增益；避免过大增益导致削波。
* noisereduce 对噪声稳定的录音效果较好（例如固定背景嗡嗡声）；对突发噪声（爆裂声）效果有限。

## 3.2 一致响度与归一化（librosa）

### 安装

```bash
pip install librosa soundfile numpy
```

### 示例

```python
import librosa
import soundfile as sf
import numpy as np

y, sr = librosa.load("input.wav", sr=None)
# 归一化到峰值的 90%
peak = np.max(np.abs(y))
if peak > 0:
    y = y / peak * 0.9
# 可选增益
y = y * 1.5
sf.write("enhanced_librosa.wav", y, sr)
```

## 3.3 AI 降噪（SpeechBrain）

### 安装

```bash
pip install speechbrain torchaudio torch
```

### 示例（保存为 `enhance_speechbrain.py`）

```python
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# 加载预训练的 spectral-mask 增强模型
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_models/mtl-mimic-voicebank"
)

# 读取音频
noisy_waveform, fs = torchaudio.load("input.wav")  # shape: (channels, samples)

# 若采样率与模型期望不符，进行重采样（模型通常期望 16 kHz）
if fs != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
    noisy_waveform = resampler(noisy_waveform)
    fs = 16000

# 添加 batch 维度: (batch, channels, samples)
noisy_batch = noisy_waveform.unsqueeze(0)

# 增强（返回 shape: batch, channels, samples）
enhanced_batch = enhancer.enhance_batch(noisy_batch)

# 恢复并保存
enhanced = enhanced_batch.squeeze(0)  # (channels, samples)
torchaudio.save("enhanced_speechbrain.wav", enhanced.cpu(), fs)
```

### 注释

* SpeechBrain 的增强模型一般在语音增强任务上效果优于传统方法；但模型下载与运行需要网络和一定显存。
* 若遇到显存不足，可尝试先用 pydub 降采样后再运行增强模型。

## 3.4 带通滤波（保留语音频段）

### 目的

保留 300–3400 Hz 的语音频段，可显著去除低频噪声与高频嘶嘶声。

### 示例函数

```python
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
```

在去噪后应用带通滤波，常可以进一步减少无关频段的干扰。

---

# 4. 全流程示例脚本（增强 -> 转写）

以下为合并示例：先执行增强（pydub + noisereduce + 带通 + 归一化），再用 Whisper 转写。

### 安装依赖

```bash
pip install -U openai-whisper torch pydub noisereduce soundfile librosa scipy numpy
# ffmpeg 仍需可用
```

### 脚本（保存为 `enhance_and_transcribe_whisper.py`）

```python
# Combined pipeline: resample -> gain -> denoise -> bandpass -> normalize -> whisper transcribe
import os
from pydub import AudioSegment
import soundfile as sf
import noisereduce as nr
import numpy as np
import whisper
from scipy.signal import butter, lfilter
import librosa

# --------- 配置 ---------
input_path = "input.wav"
temp_path = "temp_resampled.wav"
enhanced_path = "enhanced_final.wav"
# bandpass 参数
LOWCUT = 300
HIGHCUT = 3400
TARGET_SR = 16000
GAIN_DB = 10
# -----------------------

# 工具函数：带通滤波
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 1) 使用 pydub 统一为 mono + TARGET_SR，并加增益
audio = AudioSegment.from_file(input_path)
audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
audio = audio + GAIN_DB
audio.export(temp_path, format="wav")

# 2) 使用 noisereduce 去噪
data, sr = sf.read(temp_path)
if data.ndim > 1:
    data = np.mean(data, axis=1)
reduced = nr.reduce_noise(y=data, sr=sr)

# 3) 带通滤波
filtered = bandpass_filter(reduced, LOWCUT, HIGHCUT, fs=sr)

# 4) 峰值归一化（避免削波）
peak = np.max(np.abs(filtered))
if peak > 0:
    normalized = filtered / peak * 0.9
else:
    normalized = filtered

# 5) 保存增强结果
sf.write(enhanced_path, normalized, sr)
print("Enhanced audio saved to:", enhanced_path)

# 6) 使用 Whisper 转写
model = whisper.load_model("base")
result = model.transcribe(enhanced_path)
print("Transcription result:\n", result["text"])
```

---

# 5. 调试提示与常见问题

* **采样率与声道**：大多数轻量 ASR（如 Vosk）期望 16 kHz、单声道。若模型说明书要求不同，按说明调整。
* **文件格式**：若遇解码问题，使用 ffmpeg 将源文件转换为标准 WAV：

  ```bash
  ffmpeg -i input.xxx -ac 1 -ar 16000 output.wav
  ```
* **音频很小/无声**：先检查音频峰值（peak），若峰值 < 1e-4，先增益再降噪；若为静默片段则无法恢复语音信息。
* **高噪声场景**：先用 AI 降噪（SpeechBrain 或更强的模型），再使用传统滤波与归一化。
* **削波（clipping）**：避免在增益后产生超过 \[-1, 1] 的峰值。使用归一化到 0.9 或使用峰值/均方根（RMS）归一化。
* **实时/低延迟场景**：简化为只做实时增益与轻量化噪声抑制，避免批量频域处理带来的延迟。

---

# 6. 参考与资源链接

* Higgs-Audio（其他语音生成/处理模型尝试）： [https://github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)
* Whisper（OpenAI）： [https://github.com/openai/whisper](https://github.com/openai/whisper)
* Vosk： [https://alphacephei.com/vosk](https://alphacephei.com/vosk)
* SpeechBrain： [https://speechbrain.github.io](https://speechbrain.github.io)

# Datawhale AI 夏令营 — 全球AI攻防挑战赛（AIGC 技术 — 语音方向）

> 目标：
>
> * 建立稳定的 Python/Conda + `uv` 环境
> * 完成音频增强（重采样 → 增益 → 降噪 → 带通 → 归一化）批处理
> * 使用 Whisper 进行多语言识别（支持中文/英文/混合）
> * 记录 Higgs-Audio 语音生成最小示例
> * 备注安装与编译常见问题（`g++`/`cmake`/`sentencepiece`）

---

## 1) 环境搭建（Conda + uv + 独立 venv + Jupyter）

> 说明：
>
> * `conda` 负责基础 Python/系统依赖与 `uv` 安装。
> * 项目内再用 `uv venv` 创建隔离环境，避免污染 base/其他环境。
> * `ipykernel` 注册到 Jupyter，便于 Notebook 直接选择该环境。

```bash
# 创建并启用 Conda 环境（--offline 可在离线镜像场景使用）
conda create -n boson-ai-audiollm --offline
conda activate boson-ai-audiollm

# 安装 Python 与 uv（conda-forge 通道）
conda install -c conda-forge python=3.10 uv

# 拉取仓库并进入项目目录
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# 项目内创建 venv（由 uv 管理）
uv venv --python 3.10
source .venv/bin/activate

# 安装项目依赖
uv pip install -r requirements.txt
uv pip install -e .

# 为 Jupyter 安装内核（避免 pyzmq 源码编译，使用仅二进制）
uv pip install ipykernel --only-binary=pyzmq
python -m ipykernel install --user --name=higgs-audio --display-name='Environment (higgs-audio)'
```

> **可选：启动 Jupyter Lab（后台运行）**

```bash
nohup time jupyter lab --allow-root > jupy.out &
```

### 安装常见问题与提示

* **Linux 编译失败（`g++`/`cmake` 缺失）**

  * 表现：安装 `speechbrain` 时拉起 `sentencepiece` 源码编译失败。
  * 处理：

    * 方案 A（推荐）：先安装预编译 wheel

      ```bash
      pip install sentencepiece
      pip install speechbrain
      ```
    * 方案 B：补齐构建依赖再编译

      ```bash
      sudo apt-get update
      sudo apt-get install -y cmake build-essential pkg-config libprotobuf-dev protobuf-compiler
      pip install sentencepiece
      ```
    * 方案 C：使用 conda 预编译包

      ```bash
      conda install -c conda-forge sentencepiece
      pip install speechbrain
      ```
* **批量安装常用音频/ASR依赖**

  ```bash
  uv pip install openai-whisper vosk soundfile librosa pydub noisereduce speechbrain
  ```

  > 注：SpeechBrain 仅在需要 AI 增强或特定模型时安装；若不需要，可省略以避免 `sentencepiece` 编译问题。

---

## 2) 语音增强（Voice-PP）批处理

> 流程：**重采样(16k/mono) → 增益(+10dB) → 频谱降噪 → 带通(300–3400Hz) → 峰值归一化(0.9)**
> 效果：提升语音能量、抑制背景噪声、保留人声核心频段，并控制峰值防止削波。

### 2.1 数据清单读取

```python
import os
import pandas as pd

wavs_folder = '../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/'
df = pd.read_csv(wavs_folder + 'aigc_speech_generation_tasks.csv')
print(df.head())
# 期望列：
# - utt: 序号
# - reference_speech: 参考音频文件名（reference_*.wav）
# - text: 对应文本（可选）
```

### 2.2 增强函数与批处理（含详细注释）

```python
import os
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, lfilter

# --------- 参数配置 ---------
LOWCUT = 300      # 带通下限（Hz）
HIGHCUT = 3400    # 带通上限（Hz）
TARGET_SR = 16000 # 目标采样率（ASR 友好）
GAIN_DB = 10      # 增益（dB）
# ---------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    """ 设计巴特沃斯带通滤波器 """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """ 应用带通滤波 """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def enhance_wav(input_path, output_path):
    """ 单文件增强：重采样/单声道/增益 → 降噪 → 带通 → 归一化 """
    # 1) 使用 pydub 标准化：mono + TARGET_SR + 增益
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio = audio + GAIN_DB
    temp_path = output_path + "_temp.wav"
    audio.export(temp_path, format="wav")

    # 2) 频谱降噪（noisereduce）
    data, sr = sf.read(temp_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # 强制单声道
    reduced = nr.reduce_noise(y=data, sr=sr)

    # 3) 带通滤波（聚焦语音频段）
    filtered = bandpass_filter(reduced, LOWCUT, HIGHCUT, fs=sr)

    # 4) 峰值归一化（避免 clipping）
    peak = np.max(np.abs(filtered))
    normalized = filtered / peak * 0.9 if peak > 0 else filtered

    # 5) 写出并清理临时文件
    sf.write(output_path, normalized, sr)
    os.remove(temp_path)
    print(f"Enhanced saved: {output_path}")

def batch_enhance(csv_path, wavs_root):
    """ 批量增强：读取 CSV 中的 reference_*.wav 并输出 reference_enhanced_*.wav """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        wav_name = row["reference_speech"]
        input_path = os.path.join(wavs_root, wav_name)
        if os.path.isfile(input_path):
            out_name = wav_name.replace("reference_", "reference_enhanced_")
            output_path = os.path.join(wavs_root, out_name)
            enhance_wav(input_path, output_path)
        else:
            print(f"File not found: {input_path}")

# 运行（取消注释以执行）
# csv_path = os.path.join(wavs_folder, "aigc_speech_generation_tasks.csv")
# batch_enhance(csv_path, wavs_folder)
```

### 2.3 文件整理（示例）

```bash
# 增强后的文件移动至独立目录
mv AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/reference_enhanced_*.wav aisvoice-enh-wavs/

# 将任务清单一并复制（便于后续识别结果回填）
cp AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv aisvoice-enh-wavs/
```

---

## 3) 多语言识别（Whisper）

> 说明：
>
> * Whisper 原生多语种，适合包含中文/英文/混合内容。
> * 大模型（`large-v3`）精度更高；显存/内存占用更大。
> * `download_root` 可指向本地缓存路径（如 Hugging Face/ModelScope 镜像目录）。

### 3.1 可选的加速能力检测（SDPA）

```python
from transformers.utils import is_torch_sdpa_available
print(is_torch_sdpa_available())  # 若为 True，可利用 PyTorch SDPA 加速注意力
```

### 3.2 Whisper 加载与批量转写

```python
import os
import pandas as pd
import whisper
import numpy as np
import soundfile as sf

# -------- CONFIG --------
wavs_folder = "../aisvoice-enh-wavs/"
csv_path = "../aisvoice-enh-wavs/aigc_speech_generation_tasks.csv"

# Whisper 多语言模型（建议：'medium' 或 'large-v3'）
WHISPER_MODEL_NAME = 'large-v3'

# 指定模型缓存根目录（本地镜像/手动下载路径均可）
WHISPER_MODEL_PATH = "../.cache/modelscope/hub/models/openai-mirror/whisper-large-v3/"
# ------------------------

# 读取任务清单
df = pd.read_csv(csv_path)

# 简单封装
def transcribe_whisper(audio_path, model):
    result = model.transcribe(audio_path)
    return result["text"].strip()

print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, download_root=WHISPER_MODEL_PATH)

# 遍历增强后的文件并转写
whisper_results = []
for _, row in df.iterrows():
    wav_path = os.path.join(wavs_folder, f"reference_enhanced_{row['utt']}.wav")
    if not os.path.exists(wav_path):
        print(f"[WARN] Missing file: {wav_path}")
        whisper_results.append("")
        continue
    print(f"[INFO] Transcribing {wav_path} ...")
    whisper_results.append(transcribe_whisper(wav_path, whisper_model))

# 保存回 CSV
df["whisper_re"] = whisper_results
df.to_csv("aigc_speech_generation_tasks_with_asr.csv", index=False)
print("Results saved to aigc_speech_generation_tasks_with_asr.csv")
```

> 备注：
>
> * `inspect.getsource(whisper.load_model)` 可用于查看函数实现，调试加载路径与缓存逻辑。
> * 非 16 kHz 文件也可直接交给 Whisper；若前处理已统一到 16 kHz，则与前述增强流程一致。

---

## 4) Higgs-Audio 语音生成最小示例

> 说明：
>
> * 该段演示基于 Boson-AI 的生成引擎接口（**推理示例**）。
> * 需要本地提供模型权重与音频 tokenizer 目录。
> * 生成结果保存为 `output.wav`。

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio

MODEL_PATH = "../bosonai/higgs-audio-v2-generation-3B-base/"
AUDIO_TOKENIZER_PATH = "../.cache/modelscope/hub/bosonai/higgs-audio-v2-tokenizer/"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(role="system", content=system_prompt),
    Message(role="user", content=(
        "The sun rises in the east and sets in the west. "
        "This simple fact has been observed by humans for thousands of years."
    )),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)

torchaudio.save("output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

> 备注：
>
> * 命令行示例（来自脚本 `generation.py` 的典型参数约定）：
>
>   ```bash
>   python3 generation.py \
>     --transcript transcript/single_speaker/en_dl.txt \
>     --ref_audio broom_salesman \
>     --seed 12345 \
>     --out_path generation.wav
>   ```
> * 计划：以“参考音频 + 参考文本 + 目标文本”生成目标音频；若服务器存储异常或缓存目录不可写，将导致流水线中断，需要先排查磁盘或权限。

---

## 5) 实用补充与排错清单

* **FFmpeg 可用性**：

  * `pydub` 与部分模型依赖 ffmpeg；建议安装并确保 `ffmpeg -version` 可正常输出。
* **多语言与口音**：

  * Whisper `medium/large-v3` 对中英文混合与多口音更鲁棒；资源受限时可切换为 `small` 并保持前处理一致。
* **16 kHz/单声道标准化**：

  * 轻量 ASR（如 Vosk）偏好 16 kHz 单声道 PCM；前处理已满足此条件。
* **批处理吞吐**：

  * 需要更高速度时，可将增强与识别拆分为多进程/多线程；增益/带通/归一化可并行到 CPU 核心，Whisper 可在 GPU 上批跑。
* **文件命名规范**：

  * 增强输出统一以 `reference_enhanced_*.wav` 命名，便于下游识别脚本按 `utt` 自动配对。
* **`sentencepiece` 构建失败**：

  * 参见第 1 节“安装常见问题与提示”，优先采用预编译 wheel 或 conda 包。

---

## 6) 目录结构建议（示例）

```
project-root/
├─ higgs-audio/                          # 上游仓库
├─ aisvoice-enh-wavs/                    # 增强后的语音与 CSV
│  ├─ reference_enhanced_1.wav
│  ├─ reference_enhanced_2.wav
│  └─ aigc_speech_generation_tasks.csv
├─ scripts/
│  ├─ enhance_batch.py                   # 批量增强脚本
│  └─ transcribe_whisper.py              # Whisper 批量识别
└─ models_cache/                         # 本地模型缓存（可选）
```

---

### 结束

* 环境：Conda + uv + venv + Jupyter
* 增强：`pydub` + `noisereduce` + 带通 + 归一化
* 识别：Whisper 多语言（建议 `medium`/`large-v3`）
* 生成：Higgs-Audio 推理最小例
* 常见问题：`sentencepiece`/编译依赖、FFmpeg、采样率与声道标准化

> 按上述步骤逐段执行，可从原始参考音频完成增强与识别，并保留清晰的可重现脚本与结果产物。
>
> 计划是计划，服务器出问题暂时用不了了，跑得巨慢。whisper识别其实好像还有点不太准，识别出来中文繁简混合的。不过很多音频尽管经过增强去噪，让人来听都还是听不太清楚。

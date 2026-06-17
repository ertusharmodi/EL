# Eleven

A local-first voice AI companion. No cloud, no server, no API keys.

```
You speak → Whisper transcribes → qwen3:14b responds → Piper speaks back
```

---

## Prerequisites

Make sure these are installed before you begin.

| Tool | Install |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com) |
| Piper binary | See step 3 below |

---

## Setup

### 1. Pull the LLM

```bash
ollama pull qwen3:14b
```

### 2. Create a virtual environment and install Python packages

```bash
cd /Users/tusharthi.escape/Documents/Eleven
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install the Piper binary

Piper is a standalone binary. Download the macOS release and put it on your PATH:

```bash
# Download Piper for macOS (arm64 = Apple Silicon, x86_64 = Intel)
curl -L -o /tmp/piper.tar.gz \
  https://github.com/rhasspy/piper/releases/latest/download/piper_macos_aarch64.tar.gz

# Extract and move to /usr/local/bin
tar -xzf /tmp/piper.tar.gz -C /tmp
sudo mv /tmp/piper/piper /usr/local/bin/piper
sudo chmod +x /usr/local/bin/piper

# Verify
piper --help
```

> If you're on **Intel Mac**, replace `aarch64` with `x86_64` in the URL above.

### 4. Download the Piper voice model

```bash
mkdir -p voices

curl -L -o voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

curl -L -o voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## Run

```bash
source .venv/bin/activate   # if not already active
python main.py
```

Eleven will print what it's doing at each step. Speak after you see `🎙 Listening...`.
Press **Ctrl+C** to quit.

---

## Configuration

Everything you might want to change lives in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `RECORD_SECONDS` | `5` | How long Eleven listens each turn |
| `WHISPER_MODEL` | `base.en` | STT model size (`tiny.en` → `medium.en`) |
| `OLLAMA_MODEL` | `qwen3:14b` | Which Ollama model to use |
| `PIPER_VOICE` | `en_US-lessac-medium` | Piper voice model name |

---

## Project Structure

```
Eleven/
├── main.py          # Conversation loop
├── audio.py         # Mic recording + WAV playback
├── stt.py           # Faster-Whisper transcription
├── llm.py           # Ollama chat
├── tts.py           # Piper text-to-speech
├── config.py        # All settings
├── requirements.txt # Python dependencies
└── voices/          # Piper voice model files
```

---

## Troubleshooting

**"Voice model not found"** — Run the `curl` commands in step 4.

**"piper: command not found"** — Make sure `/usr/local/bin/piper` exists and is executable. Re-run step 3.

**Ollama connection error** — Make sure Ollama is running: `ollama serve` (in a separate terminal).

**No audio input** — Go to System Settings → Privacy & Security → Microphone and allow Terminal access.

**Whisper model downloads on first run** — That's expected. It's ~140MB and happens once.

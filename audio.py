# audio.py — Microphone recording, VAD, and WAV playback.
#
# VAD engine: Silero VAD (neural, ~2MB model).
# Falls back to energy-based VAD if silero-vad is not installed.

import sounddevice as sd
import soundfile as sf
import numpy as np

import config

# ── Silero model singleton ──────────────────────────────────────────────────
# Loaded once on first call to record_vad(); cached here to avoid reloading
# between conversation turns.
_silero_model = None
_silero_utils = None

def _get_silero_model():
    """Load and cache the Silero VAD model. Returns (model, get_speech_ts) or None."""
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return _silero_model, _silero_utils
    try:
        from silero_vad import load_silero_vad
        import torch
        model = load_silero_vad()
        model.eval()
        _silero_model = model
        _silero_utils = torch
        print("  ✅  [VAD] Silero VAD model loaded.")
        return model, _silero_utils
    except Exception as exc:
        print(f"  ⚠️   [VAD] Could not load Silero VAD ({exc}). Falling back to energy VAD.")
        return None, None


# ── Noise floor calibration ─────────────────────────────────────────────────

# Module-level noise floor (RMS, float32 normalised). Updated by calibrate_noise().
_noise_floor: float = 0.0
_noise_floor_calibrated: bool = False


def calibrate_noise() -> float:
    """
    Record VAD_CALIBRATION_SECONDS of ambient audio and compute the RMS noise floor.

    Should be called once at startup before the conversation loop.
    Stores the result in _noise_floor and prints diagnostics.
    Returns the noise floor RMS value.
    """
    global _noise_floor, _noise_floor_calibrated

    duration = config.VAD_CALIBRATION_SECONDS
    print(f"\n  🔇  Calibrating noise floor ({duration:.1f}s) — please stay quiet...")

    frames = sd.rec(
        frames=int(duration * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="int16",
    )
    sd.wait()

    audio = frames.astype(np.float32) / 32768.0
    rms_per_chunk = []
    chunk = int(config.VAD_CHUNK_DURATION * config.SAMPLE_RATE)
    for i in range(0, len(audio) - chunk + 1, chunk):
        rms_per_chunk.append(float(np.sqrt(np.mean(audio[i:i + chunk] ** 2))))

    if rms_per_chunk:
        mean_rms = float(np.mean(rms_per_chunk))
        std_rms  = float(np.std(rms_per_chunk))
        floor    = mean_rms + 2 * std_rms   # 2σ above mean = conservative floor
    else:
        mean_rms = std_rms = floor = 0.0

    _noise_floor = floor
    _noise_floor_calibrated = True

    print(
        f"  🔇  [VAD] Noise floor: RMS={floor:.5f}  "
        f"(mean={mean_rms:.5f}, 2σ={2*std_rms:.5f})\n"
    )
    return floor


# ── Fixed-length recording (legacy) ────────────────────────────────────────

def record(output_path: str = config.RECORDED_WAV) -> str:
    """
    Record RECORD_SECONDS of audio from the default microphone.
    Saves as a 16kHz mono WAV file. Returns the output path.
    """
    print(f"  🎙  Listening for {config.RECORD_SECONDS} seconds...")

    frames = sd.rec(
        frames=int(config.RECORD_SECONDS * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="int16",
    )
    sd.wait()

    sf.write(output_path, frames, config.SAMPLE_RATE)
    return output_path


# ── VAD recording ───────────────────────────────────────────────────────────

def record_vad(output_path: str = config.RECORDED_WAV) -> str:
    """
    Record from the microphone using Silero VAD to detect speech/silence.

    Algorithm:
      1. Silero VAD scores each 32ms chunk with speech_prob (0–1).
      2. State machine: PRE_SPEECH → IN_SPEECH → IN_SILENCE.
         - Only transitions to IN_SPEECH after VAD_MIN_SPEECH_DURATION of
           consecutive speech (filters keyboard clicks, brief noise spikes).
         - Silence timer starts only after confirmed speech ends.
         - Stops after VAD_SILENCE_TIMEOUT seconds of continuous silence.
      3. Falls back to energy-based VAD if Silero is unavailable.

    Returns the output path.
    """
    import queue
    import time
    import torch

    use_silero = (config.VAD_ENGINE == "silero")
    model = torch_mod = None

    if use_silero:
        model, torch_mod = _get_silero_model()
        if model is None:
            use_silero = False   # fall back to energy

    chunk_size = int(config.VAD_CHUNK_DURATION * config.SAMPLE_RATE)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(indata, frames, time_info, status):
        q.put(indata.copy())

    # ── State ───────────────────────────────────────────────────────────────
    all_chunks      = []
    total_samples   = 0
    max_samples     = int(config.VAD_MAX_DURATION   * config.SAMPLE_RATE)
    min_samples     = int(config.VAD_MIN_DURATION   * config.SAMPLE_RATE)
    min_speech_samp = int(config.VAD_MIN_SPEECH_DURATION * config.SAMPLE_RATE)

    # State machine states
    STATE_PRE_SPEECH = "PRE_SPEECH"
    STATE_IN_SPEECH  = "IN_SPEECH"
    STATE_IN_SILENCE = "IN_SILENCE"

    state            = STATE_PRE_SPEECH
    speech_start_t   = None    # monotonic time when first speech chunk seen
    speech_samples   = 0       # consecutive speech-classified samples
    silence_start    = None    # monotonic time when silence began
    stop_reason      = "max_duration"

    # ── Startup diagnostics ─────────────────────────────────────────────────
    engine_label = "Silero VAD" if use_silero else "Energy VAD (fallback)"
    print(f"\n  🎙  Listening... (speak now — stops after {config.VAD_SILENCE_TIMEOUT}s silence)")
    if config.VAD_DEBUG:
        if use_silero:
            print(
                f"  🔧  [VAD] engine={engine_label}  "
                f"threshold={config.VAD_SILERO_THRESHOLD}  "
                f"min_speech={config.VAD_MIN_SPEECH_DURATION}s  "
                f"silence_timeout={config.VAD_SILENCE_TIMEOUT}s  "
                f"max={config.VAD_MAX_DURATION}s"
            )
        else:
            threshold = max(_noise_floor * 1.5, 0.01) if _noise_floor else 0.01
            print(
                f"  🔧  [VAD] engine={engine_label}  "
                f"threshold={threshold:.5f}  "
                f"noise_floor={_noise_floor:.5f}  "
                f"silence_timeout={config.VAD_SILENCE_TIMEOUT}s"
            )

    # ── Recording loop ──────────────────────────────────────────────────────
    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        dtype="int16",
        blocksize=chunk_size,
        callback=_callback,
    ):
        while True:
            chunk = q.get()
            all_chunks.append(chunk)
            total_samples += len(chunk)
            elapsed = total_samples / config.SAMPLE_RATE

            # ── Classify chunk ───────────────────────────────────────────
            audio_f32 = chunk.astype(np.float32) / 32768.0
            rms       = float(np.sqrt(np.mean(audio_f32 ** 2)))

            if use_silero:
                tensor     = torch.from_numpy(audio_f32.squeeze())
                speech_prob = float(model(tensor, config.SAMPLE_RATE).item())
                is_speech   = speech_prob >= config.VAD_SILERO_THRESHOLD
            else:
                # Energy fallback: dynamic threshold = noise_floor * 1.5 or 0.01
                threshold   = max(_noise_floor * 1.5, 0.01) if _noise_floor else 0.01
                speech_prob = rms / (threshold + 1e-9)   # pseudo-confidence ratio
                is_speech   = rms >= threshold

            # ── Per-chunk diagnostic ─────────────────────────────────────
            if config.VAD_DEBUG:
                bar  = "█" * min(int(speech_prob * 20), 20) if use_silero else "█" * min(int(rms * 500), 20)
                prob_str = f"prob={speech_prob:.3f}" if use_silero else f"RMS={rms:.5f}"
                print(
                    f"    {elapsed:5.1f}s  {prob_str}  RMS={rms:.5f}  "
                    f"{'🗣' if is_speech else '·'} {bar}  [{state}]"
                )

            # ── VAD state machine ────────────────────────────────────────
            now = time.monotonic()

            if state == STATE_PRE_SPEECH:
                if is_speech:
                    speech_samples += len(chunk)
                    if speech_start_t is None:
                        speech_start_t = now
                    # Only confirm speech after min_speech_duration
                    if speech_samples >= min_speech_samp:
                        state = STATE_IN_SPEECH
                        print(
                            f"  🟢  [VAD] Speech started at {elapsed:.1f}s  "
                            f"{'prob=' + str(round(speech_prob, 3)) if use_silero else 'RMS=' + str(round(rms, 5))}"
                        )
                else:
                    # Noise spike — reset consecutive speech counter
                    speech_samples = 0
                    speech_start_t = None

            elif state == STATE_IN_SPEECH:
                if is_speech:
                    # Still speaking — cancel any stale silence timer
                    if silence_start is not None:
                        if config.VAD_DEBUG:
                            print(
                                f"  🔄  [VAD] Silence timer reset at {elapsed:.1f}s "
                                f"(speech resumed after {now - silence_start:.2f}s)"
                            )
                        silence_start = None
                    state = STATE_IN_SPEECH   # stay
                else:
                    # First silent chunk after speech
                    if silence_start is None:
                        silence_start = now
                        print(
                            f"  🔴  [VAD] Speech ended at {elapsed:.1f}s  "
                            f"— silence timer started"
                        )
                    state = STATE_IN_SILENCE

            elif state == STATE_IN_SILENCE:
                if is_speech:
                    # Speech resumed — cancel silence timer, go back to IN_SPEECH
                    silence_duration = now - silence_start if silence_start else 0
                    if config.VAD_DEBUG:
                        print(
                            f"  🔄  [VAD] Silence timer reset at {elapsed:.1f}s "
                            f"(speech resumed after {silence_duration:.2f}s)"
                        )
                    silence_start = None
                    state = STATE_IN_SPEECH
                else:
                    # Still silent — check timeout
                    silence_duration = now - silence_start
                    if config.VAD_DEBUG:
                        print(
                            f"    {'':>6}  silence={silence_duration:.2f}s / {config.VAD_SILENCE_TIMEOUT}s"
                        )
                    if (total_samples >= min_samples and
                            silence_duration >= config.VAD_SILENCE_TIMEOUT):
                        stop_reason = "silence_timeout"
                        print(
                            f"  ⏹   [VAD] Silence timeout after {silence_duration:.2f}s "
                            f"— stopping recording."
                        )
                        break

            # ── Hard cap ─────────────────────────────────────────────────
            if total_samples >= max_samples:
                stop_reason = "max_duration"
                print(f"  ⚠️   [VAD] Max duration ({config.VAD_MAX_DURATION}s) reached — stopping.")
                break

    frames   = np.concatenate(all_chunks, axis=0)
    duration = total_samples / config.SAMPLE_RATE
    print(f"  ✅  Recorded {duration:.1f}s  [stop_reason={stop_reason}]")
    sf.write(output_path, frames, config.SAMPLE_RATE)
    return output_path


# ── Playback ────────────────────────────────────────────────────────────────

def play(wav_path: str) -> None:
    """
    Play a WAV file through the default speaker.
    Blocks until playback is complete.
    """
    data, samplerate = sf.read(wav_path, dtype="float32")
    sd.play(data, samplerate, blocksize=config.PLAYBACK_BLOCKSIZE)
    sd.wait()

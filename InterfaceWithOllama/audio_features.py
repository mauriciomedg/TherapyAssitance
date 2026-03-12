from pathlib import Path
import math
import numpy as np
import librosa


def compute_pause_stats(segments: list[dict], long_pause_threshold: float = 1.5) -> dict:
    if not segments:
        return {
            "speech_duration_sec": 0.0,
            "audio_duration_sec": 0.0,
            "speech_ratio": 0.0,
            "silence_ratio": 0.0,
            "pause_count": 0,
            "avg_pause_sec": 0.0,
            "long_pause_count": 0,
            "avg_segment_duration_sec": 0.0,
        }

    speech_duration = 0.0
    pauses = []

    for seg in segments:
        speech_duration += max(0.0, seg["end"] - seg["start"])

    for i in range(1, len(segments)):
        prev_end = segments[i - 1]["end"]
        curr_start = segments[i]["start"]
        pause = max(0.0, curr_start - prev_end)
        if pause > 0:
            pauses.append(pause)

    audio_duration = max(seg["end"] for seg in segments)
    silence_duration = max(0.0, audio_duration - speech_duration)

    avg_pause = sum(pauses) / len(pauses) if pauses else 0.0
    avg_segment_duration = speech_duration / len(segments) if segments else 0.0
    long_pause_count = sum(1 for p in pauses if p >= long_pause_threshold)

    speech_ratio = speech_duration / audio_duration if audio_duration > 0 else 0.0
    silence_ratio = silence_duration / audio_duration if audio_duration > 0 else 0.0

    return {
        "speech_duration_sec": round(speech_duration, 3),
        "audio_duration_sec": round(audio_duration, 3),
        "speech_ratio": round(speech_ratio, 3),
        "silence_ratio": round(silence_ratio, 3),
        "pause_count": len(pauses),
        "avg_pause_sec": round(avg_pause, 3),
        "long_pause_count": long_pause_count,
        "avg_segment_duration_sec": round(avg_segment_duration, 3),
    }


def compute_energy_features(audio_path: Path) -> dict:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    if y.size == 0:
        return {"avg_energy_rms": 0.0, "std_energy_rms": 0.0}

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    return {
        "avg_energy_rms": round(float(np.mean(rms)), 6),
        "std_energy_rms": round(float(np.std(rms)), 6),
    }


def classify_energy_band(avg_rms: float) -> str:
    if avg_rms < 0.01:
        return "very low"
    if avg_rms < 0.03:
        return "low"
    if avg_rms < 0.07:
        return "moderate"
    return "high"


def build_audio_behavior_summary(feature_dict: dict) -> str:
    energy_band = classify_energy_band(feature_dict.get("avg_energy_rms", 0.0))

    lines = [
        f"- Speech occupied about {round(feature_dict.get('speech_ratio', 0.0) * 100)}% of the audio.",
        f"- Silence occupied about {round(feature_dict.get('silence_ratio', 0.0) * 100)}% of the audio.",
        f"- There were {feature_dict.get('pause_count', 0)} pauses between speaking segments.",
        f"- Average pause duration was {feature_dict.get('avg_pause_sec', 0.0)} seconds.",
        f"- {feature_dict.get('long_pause_count', 0)} pauses were longer than 1.5 seconds.",
        f"- Average vocal energy was {energy_band}.",
    ]
    return "\n".join(lines)


def extract_audio_behavior_features(audio_path: Path, segments: list[dict]) -> dict:
    pause_stats = compute_pause_stats(segments)
    energy_stats = compute_energy_features(audio_path)

    features = {**pause_stats, **energy_stats}
    features["audio_behavior_summary"] = build_audio_behavior_summary(features)
    return features
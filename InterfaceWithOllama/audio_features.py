from pathlib import Path
import numpy as np
import librosa


def classify_energy_band(avg_rms: float) -> str:
    if avg_rms < 0.01:
        return "very low"
    if avg_rms < 0.03:
        return "low"
    if avg_rms < 0.07:
        return "moderate"
    return "high"


def compute_global_energy(audio_path: Path) -> dict:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    if y.size == 0:
        return {
            "avg_energy_rms": 0.0,
            "std_energy_rms": 0.0,
            "sr": sr,
            "waveform": y,
        }

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    return {
        "avg_energy_rms": float(np.mean(rms)),
        "std_energy_rms": float(np.std(rms)),
        "sr": sr,
        "waveform": y,
    }


def compute_segment_energy(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> float:
    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(y), int(end_sec * sr))

    if end_idx <= start_idx:
        return 0.0

    segment = y[start_idx:end_idx]
    if segment.size == 0:
        return 0.0

    rms = np.sqrt(np.mean(segment ** 2))
    return float(rms)


def describe_pause_before(pause_sec: float) -> str:
    if pause_sec <= 0.05:
        return "no noticeable pause before speaking"
    if pause_sec < 0.75:
        return "a brief pause before speaking"
    if pause_sec < 1.5:
        return "a moderate pause before speaking"
    return "a longer pause before speaking"


def describe_relative_energy(local_rms: float, global_avg_rms: float) -> str:
    if global_avg_rms <= 1e-8:
        return "unknown vocal energy"

    ratio = local_rms / global_avg_rms

    if ratio < 0.65:
        return "lower vocal energy than earlier in the session"
    if ratio > 1.35:
        return "higher vocal energy than earlier in the session"
    return "similar vocal energy to the rest of the session"


def is_salient_segment(pause_before: float, local_rms: float, global_avg_rms: float) -> bool:
    if pause_before >= 1.5:
        return True

    if global_avg_rms > 1e-8:
        ratio = local_rms / global_avg_rms
        if ratio < 0.65 or ratio > 1.35:
            return True

    return False


def extract_segment_audio_features(audio_path: Path, segments: list[dict]) -> dict:
    global_audio = compute_global_energy(audio_path)
    y = global_audio["waveform"]
    sr = global_audio["sr"]
    global_avg_rms = global_audio["avg_energy_rms"]

    enriched_segments = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]

        if i == 0:
            pause_before = 0.0
        else:
            prev_end = segments[i - 1]["end"]
            pause_before = max(0.0, start - prev_end)

        local_rms = compute_segment_energy(y, sr, start, end)
        pause_desc = describe_pause_before(pause_before)
        energy_desc = describe_relative_energy(local_rms, global_avg_rms)
        salient = is_salient_segment(pause_before, local_rms, global_avg_rms)

        enriched_segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "pause_before_sec": round(pause_before, 3),
                "local_energy_rms": round(local_rms, 6),
                "pause_description": pause_desc,
                "energy_description": energy_desc,
                "is_salient": salient,
            }
        )

    return {
        "global_avg_energy_rms": round(global_avg_rms, 6),
        "global_energy_band": classify_energy_band(global_avg_rms),
        "segments": enriched_segments,
    }


def build_segment_context_for_prompt(segment_audio_data: dict) -> str:
    lines = []

    for seg in segment_audio_data["segments"]:
        text = seg["text"].strip()
        if not text:
            continue

        lines.append(f"Segment [{seg['start']:.2f}s - {seg['end']:.2f}s]")
        lines.append(f"Text: {text}")

        if seg["is_salient"]:
            lines.append("Relevant speech behavior:")
            lines.append(f"- {seg['pause_description']}")
            lines.append(f"- {seg['energy_description']}")
        else:
            lines.append("Relevant speech behavior:")
            lines.append("- no strongly notable speech behavior in this segment")

        lines.append("")

    return "\n".join(lines).strip()
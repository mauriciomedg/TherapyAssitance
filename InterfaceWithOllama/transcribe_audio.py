from pathlib import Path
import json
import sys

from faster_whisper import WhisperModel


# Change this if needed:
MODEL_SIZE = "large-v3"   # try: "small", "medium", "large-v3"
#DEVICE = "cuda"         # use "cpu" if you do not want GPU
#COMPUTE_TYPE = "float16"  # for many NVIDIA GPUs; use "int8" on CPU

DEVICE = "cpu"
COMPUTE_TYPE = "int8"

#import os
#print(os.environ.get("PATH", ""))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def transcribe_audio(audio_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )

    segment_list = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        segment_list.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": text,
            }
        )

    return info, segment_list


def save_transcript_json(path: Path, info, segments) -> None:
    ensure_parent_dir(path)

    payload = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": segments,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_transcript_txt(path: Path, segments) -> None:
    ensure_parent_dir(path)

    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_plain_text(path: Path, segments) -> None:
    ensure_parent_dir(path)
    plain_text = "\n".join(seg["text"] for seg in segments)
    path.write_text(plain_text, encoding="utf-8")


def main() -> int:
    input_audio = Path("input/Recording.m4a")
    output_json = Path("output/transcript.json")
    output_txt = Path("output/transcript_timestamps.txt")
    output_plain = Path("output/transcript.txt")

    if len(sys.argv) > 1:
        input_audio = Path(sys.argv[1])

    try:
        info, segments = transcribe_audio(input_audio)

        save_transcript_json(output_json, info, segments)
        save_transcript_txt(output_txt, segments)
        save_plain_text(output_plain, segments)

        print("\n=== TRANSCRIPTION COMPLETE ===\n")
        print(f"Detected language: {getattr(info, 'language', 'unknown')}")
        print(f"Plain transcript:   {output_plain.resolve()}")
        print(f"Timestamped text:   {output_txt.resolve()}")
        print(f"JSON transcript:    {output_json.resolve()}")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
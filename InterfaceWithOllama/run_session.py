from pathlib import Path
import json
import sys
import requests

from faster_whisper import WhisperModel
from audio_features import extract_segment_audio_features, build_segment_context_for_prompt

# =========================
# CONFIG
# =========================

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cpu"         # change to "cuda" later if desired
WHISPER_COMPUTE_TYPE = "int8"  # good for CPU

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "qwen3:8b"

DEFAULT_CLEANUP_PROMPT_TEMPLATE = """
Está corrigiendo una transcripción automática de una sesión de terapia.

Contexto:
La transcripción puede contener términos frecuentes como: paciente, terapeuta, ansiedad, reuniones, trabajo, pensamientos automáticos, juicio, respiración, dormir.

Reglas:
- Corrija errores obvios de transcripción y errores fonéticos.
- Preserve el significado original.
- No agregue información nueva.
- No resuma.
- Mantenga el texto en el mismo idioma que la transcripción.
- Si una palabra o frase parece claramente mal transcrita, reemplácela por la formulación más probable.
- Mantenga claras y naturales las referencias al paciente y al terapeuta.
- Elimine palabras absurdas o nombres erróneos cuando claramente sean errores de transcripción.
- Devuelva únicamente la transcripción corregida.

Transcripción:
{TRANSCRIPTION}
"""
DEFAULT_SUMMARY_PROMPT_TEMPLATE = """
You are assisting a licensed psychologist.

The following material comes from a therapy session.
It may contain Spanish, French, or English speech.

You are given transcript segments aligned with speech behavior observations.
Each segment may include:
- the spoken text
- whether there was a longer pause before speaking
- whether vocal energy was lower or higher than the rest of the session

Your task is to write a concise summary in English.

Instructions:
- Base the summary only on the provided segment text and aligned speech behavior observations.
- Integrate speech behavior naturally into the sentence describing the relevant content when useful.
- Only mention speech behavior when it adds context to an important part of the discussion.
- Do not list behavior cues separately at the end.
- Do not include raw statistics, percentages, or timestamps.
- Do not infer diagnosis, hidden emotions, or psychological states from speech behavior alone.
- Use neutral phrasing such as "The patient reports..." and "The therapist explores...".
- Do not add facts or conclusions not supported by the provided material.
- Write the summary entirely in English.

Aligned session segments:
{TRANSCRIPTION}
"""

# =========================
# HELPERS
# =========================
import re

def normalize_for_repeat_detection(text: str) -> str:
    """
    Normalize text for fuzzy duplicate comparison.
    Lowercase, collapse spaces, and remove most punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúüñàèìòùâêîôûç]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_near_duplicate(a: str, b: str) -> bool:
    """
    Returns True if b is effectively the same as a, or a slightly longer
    completion of a, which is common in Whisper loop artifacts.

    Examples:
    - 'y me fui a comprar un bag'
      vs 'y me fui a comprar un bagel'  -> True
    - exact repeated phrase             -> True
    """
    na = normalize_for_repeat_detection(a)
    nb = normalize_for_repeat_detection(b)

    if not na or not nb:
        return False

    # Exact normalized match
    if na == nb:
        return True

    # One is a prefix of the other (partial-word / incremental decoding artifact)
    if na.startswith(nb) or nb.startswith(na):
        shorter = min(len(na), len(nb))
        longer = max(len(na), len(nb))
        # Avoid collapsing unrelated very short strings
        if shorter >= 10 and longer <= int(shorter * 1.6):
            return True

    return False


def remove_consecutive_repetition_loops(text: str) -> str:
    """
    Removes consecutive Whisper-style repetition loops line-by-line.
    Keeps the most complete version in runs like:
      '... comprar un bag'
      '... comprar un bagel'
      '... comprar un bagel'
    """
    # Split into sentence-like chunks while preserving order.
    # This is intentionally simple and robust for messy transcripts.
    chunks = re.split(r'(?<=[\.\!\?\n])\s+|,\s+', text)
    chunks = [c.strip() for c in chunks if c and c.strip()]

    if not chunks:
        return text.strip()

    cleaned = []
    current_best = chunks[0]

    for chunk in chunks[1:]:
        if is_near_duplicate(current_best, chunk):
            # Keep the more complete / longer one
            if len(normalize_for_repeat_detection(chunk)) > len(normalize_for_repeat_detection(current_best)):
                current_best = chunk
        else:
            cleaned.append(current_best)
            current_best = chunk

    cleaned.append(current_best)

    # Rebuild as readable text
    result = ". ".join(c.rstrip(".,;: ") for c in cleaned if c.strip())
    result = re.sub(r"\s+", " ", result).strip()

    if result and result[-1] not in ".!?":
        result += "."

    return result

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


def call_ollama(prompt: str, model: str = OLLAMA_MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    return data["response"].strip()


def build_prompt(prompt_template: str, transcript_text: str, audio_features_text: str = "") -> str:
    if "{TRANSCRIPTION}" not in prompt_template:
        raise ValueError("Prompt template must contain the placeholder {TRANSCRIPTION}")

    prompt = prompt_template.replace("{TRANSCRIPTION}", transcript_text)

    if "{AUDIO_FEATURES}" in prompt:
        prompt = prompt.replace("{AUDIO_FEATURES}", audio_features_text)

    return prompt
# =========================
# WHISPER / TRANSCRIPTION
# =========================

def transcribe_audio(audio_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
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


def save_transcript_timestamps(path: Path, segments) -> None:
    ensure_parent_dir(path)
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plain_transcript(path: Path, segments) -> None:
    ensure_parent_dir(path)
    plain_text = "\n".join(seg["text"] for seg in segments)
    path.write_text(plain_text, encoding="utf-8")


# =========================
# OLLAMA STAGES
# =========================

def clean_transcript(raw_transcript: str, cleanup_prompt_template: str) -> tuple[str, str]:
    prompt = build_prompt(cleanup_prompt_template, raw_transcript)
    cleaned = call_ollama(prompt)
    return cleaned, prompt


def summarize_transcript(
    cleaned_transcript: str,
    summary_prompt_template: str,
    audio_features_text: str = "",
) -> tuple[str, str]:
    prompt = build_prompt(summary_prompt_template, cleaned_transcript, audio_features_text)
    summary = call_ollama(prompt)
    return summary, prompt

# =========================
# PIPELINES
# =========================

def run_text_pipeline(
    raw_transcript_text: str,
    cleanup_prompt_template: str = DEFAULT_CLEANUP_PROMPT_TEMPLATE,
    summary_prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
    audio_features_text: str = "",
) -> dict:
    output_dir = Path("output")
    raw_transcript_plain_path = output_dir / "transcript_raw.txt"
    clean_transcript_path = output_dir / "transcript_clean.txt"
    cleanup_prompt_log_path = output_dir / "prompt_cleanup.txt"
    summary_prompt_log_path = output_dir / "prompt_summary.txt"
    summary_path = output_dir / "summary.txt"

    raw_transcript_text = raw_transcript_text.strip()
    if not raw_transcript_text:
        raise ValueError("Raw transcript is empty.")

    ensure_parent_dir(raw_transcript_plain_path)
    raw_transcript_plain_path.write_text(raw_transcript_text, encoding="utf-8")

    precleaned_transcript = remove_consecutive_repetition_loops(raw_transcript_text)

    cleaned_transcript, cleanup_prompt = clean_transcript(
        precleaned_transcript,
        cleanup_prompt_template,
    )

    clean_transcript_path.write_text(cleaned_transcript, encoding="utf-8")
    cleanup_prompt_log_path.write_text(cleanup_prompt, encoding="utf-8")

    summary, summary_prompt = summarize_transcript(
        cleaned_transcript,
        summary_prompt_template,
        audio_features_text="",
    )
    summary_path.write_text(summary, encoding="utf-8")
    summary_prompt_log_path.write_text(summary_prompt, encoding="utf-8")

    return {
        "detected_language": "from_text_input",
        "raw_transcript": raw_transcript_text,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
        "cleanup_prompt_used": cleanup_prompt,
        "summary_prompt_used": summary_prompt,
        "raw_transcript_path": str(raw_transcript_plain_path.resolve()),
        "clean_transcript_path": str(clean_transcript_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def run_summary_only_pipeline(
    cleaned_transcript_text: str,
    summary_prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
) -> dict:
    output_dir = Path("output")
    clean_transcript_path = output_dir / "transcript_clean.txt"
    summary_prompt_log_path = output_dir / "prompt_summary.txt"
    summary_path = output_dir / "summary.txt"

    cleaned_transcript_text = cleaned_transcript_text.strip()
    if not cleaned_transcript_text:
        raise ValueError("Cleaned transcript is empty.")

    ensure_parent_dir(clean_transcript_path)
    clean_transcript_path.write_text(cleaned_transcript_text, encoding="utf-8")

    summary, summary_prompt = summarize_transcript(
        cleaned_transcript_text,
        summary_prompt_template,
    )
    summary_path.write_text(summary, encoding="utf-8")
    summary_prompt_log_path.write_text(summary_prompt, encoding="utf-8")

    return {
        "detected_language": "from_cleaned_text_input",
        "raw_transcript": "",
        "cleaned_transcript": cleaned_transcript_text,
        "summary": summary,
        "cleanup_prompt_used": "",
        "summary_prompt_used": summary_prompt,
        "raw_transcript_path": "",
        "clean_transcript_path": str(clean_transcript_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def run_pipeline(
    audio_path: Path,
    cleanup_prompt_template: str = DEFAULT_CLEANUP_PROMPT_TEMPLATE,
    summary_prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
) -> dict:
    output_dir = Path("output")
    raw_transcript_json_path = output_dir / "transcript.json"
    raw_transcript_timestamps_path = output_dir / "transcript_timestamps.txt"
    raw_transcript_plain_path = output_dir / "transcript_raw.txt"
    segment_audio_features_path = output_dir / "segment_audio_features.json"
    aligned_prompt_context_path = output_dir / "aligned_segments_for_summary.txt"
    clean_transcript_path = output_dir / "transcript_clean.txt"
    cleanup_prompt_log_path = output_dir / "prompt_cleanup.txt"
    summary_prompt_log_path = output_dir / "prompt_summary.txt"
    summary_path = output_dir / "summary.txt"

    info, segments = transcribe_audio(audio_path)

    save_transcript_json(raw_transcript_json_path, info, segments)
    save_transcript_timestamps(raw_transcript_timestamps_path, segments)
    save_plain_transcript(raw_transcript_plain_path, segments)

    raw_transcript_text = raw_transcript_plain_path.read_text(encoding="utf-8").strip()
    if not raw_transcript_text:
        raise ValueError("Raw transcript is empty after transcription.")

    precleaned_transcript = remove_consecutive_repetition_loops(raw_transcript_text)

    cleaned_transcript, cleanup_prompt = clean_transcript(
        precleaned_transcript,
        cleanup_prompt_template,
    )

    clean_transcript_path.write_text(cleaned_transcript, encoding="utf-8")
    cleanup_prompt_log_path.write_text(cleanup_prompt, encoding="utf-8")

    # Replace original segment text with cleaned sentences approximately line by line when possible.
    # For V2.5 we keep original segment texts for timing alignment and use cleaned full transcript separately.
    segment_audio_data = extract_segment_audio_features(audio_path, segments)
    segment_audio_features_path.write_text(
        json.dumps(segment_audio_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aligned_segment_context = build_segment_context_for_prompt(segment_audio_data)
    aligned_prompt_context_path.write_text(aligned_segment_context, encoding="utf-8")

    summary, summary_prompt = summarize_transcript(
        aligned_segment_context,
        summary_prompt_template,
    )
    summary_path.write_text(summary, encoding="utf-8")
    summary_prompt_log_path.write_text(summary_prompt, encoding="utf-8")

    return {
        "detected_language": getattr(info, "language", "unknown"),
        "raw_transcript": raw_transcript_text,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
        "aligned_segment_context": aligned_segment_context,
        "cleanup_prompt_used": cleanup_prompt,
        "summary_prompt_used": summary_prompt,
        "raw_transcript_path": str(raw_transcript_plain_path.resolve()),
        "clean_transcript_path": str(clean_transcript_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "segment_audio_features_path": str(segment_audio_features_path.resolve()),
        "aligned_prompt_context_path": str(aligned_prompt_context_path.resolve()),
    }


# =========================
# CLI
# =========================

def main() -> int:
    input_audio = Path("input/session.m4a")

    if len(sys.argv) > 1:
        input_audio = Path(sys.argv[1])

    try:
        result = run_pipeline(input_audio)

        print("=== DETECTED TRANSCRIPT LANGUAGE ===")
        print(result["detected_language"])

        print("\n=== RAW TRANSCRIPT ===\n")
        print(result["raw_transcript"])

        print("\n=== CLEANED TRANSCRIPT ===\n")
        print(result["cleaned_transcript"])

        print("\n=== GENERATED SUMMARY ===\n")
        print(result["summary"])
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
from pathlib import Path
import json
import sys
import requests

from faster_whisper import WhisperModel


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

The following transcript may contain Spanish, French, or English speech from a therapy session.
Your task is to understand the transcript and write the final summary in English.

The transcript may contain statements from both the patient and the therapist.

Rules:
- Use only information explicitly present in the transcript.
- Do not add facts, interpretations, or conclusions.
- Do not make a diagnosis.
- Do not write sentences about what was not mentioned.
- If a detail is not explicitly stated, omit it.
- Use neutral language such as "The patient reports..." or "The therapist explores..."
- Write the summary entirely in English.
- Limit the summary to 80–120 words.

Transcript:
{TRANSCRIPTION}
"""


# =========================
# HELPERS
# =========================

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


def build_prompt(prompt_template: str, transcript_text: str) -> str:
    if "{TRANSCRIPTION}" not in prompt_template:
        raise ValueError("Prompt template must contain the placeholder {TRANSCRIPTION}")
    return prompt_template.replace("{TRANSCRIPTION}", transcript_text)


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


def summarize_transcript(cleaned_transcript: str, summary_prompt_template: str) -> tuple[str, str]:
    prompt = build_prompt(summary_prompt_template, cleaned_transcript)
    summary = call_ollama(prompt)
    return summary, prompt


# =========================
# PIPELINES
# =========================

def run_text_pipeline(
    raw_transcript_text: str,
    cleanup_prompt_template: str = DEFAULT_CLEANUP_PROMPT_TEMPLATE,
    summary_prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
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

    cleaned_transcript, cleanup_prompt = clean_transcript(
        raw_transcript_text,
        cleanup_prompt_template,
    )
    clean_transcript_path.write_text(cleaned_transcript, encoding="utf-8")
    cleanup_prompt_log_path.write_text(cleanup_prompt, encoding="utf-8")

    summary, summary_prompt = summarize_transcript(
        cleaned_transcript,
        summary_prompt_template,
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

    info, segments = transcribe_audio(audio_path)

    save_transcript_json(raw_transcript_json_path, info, segments)
    save_transcript_timestamps(raw_transcript_timestamps_path, segments)
    save_plain_transcript(raw_transcript_plain_path, segments)

    raw_transcript_text = raw_transcript_plain_path.read_text(encoding="utf-8").strip()
    if not raw_transcript_text:
        raise ValueError("Raw transcript is empty after transcription.")

    result = run_text_pipeline(
        raw_transcript_text,
        cleanup_prompt_template=cleanup_prompt_template,
        summary_prompt_template=summary_prompt_template,
    )

    result["detected_language"] = getattr(info, "language", "unknown")
    return result


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
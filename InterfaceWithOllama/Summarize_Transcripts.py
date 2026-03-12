from pathlib import Path
import json
import sys
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

PROMPT_TEMPLATE = """
You are assisting a licensed psychologist.

The following transcript may contain Spanish speech from a therapy session. 
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


def load_transcript(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transcript file is empty: {path}")
    return text


def build_prompt(transcript: str) -> str:
    return PROMPT_TEMPLATE.replace("{TRANSCRIPTION}", transcript)


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data, ensure_ascii=False, indent=2)}")

    return data["response"].strip()


def save_summary(path: Path, summary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")


def main() -> int:
    input_path = Path("output/transcript.txt")
    output_path = Path("output/summary.txt")

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    try:
        transcript = load_transcript(input_path)
        prompt = build_prompt(transcript)
        summary = call_ollama(prompt)

        save_summary(output_path, summary)

        print("\n=== GENERATED SUMMARY ===\n")
        print(summary)
        print(f"\nSaved to: {output_path.resolve()}")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
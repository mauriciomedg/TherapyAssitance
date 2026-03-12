import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from run_session import (
    run_pipeline,
    DEFAULT_CLEANUP_PROMPT_TEMPLATE,
    DEFAULT_SUMMARY_PROMPT_TEMPLATE,
)


class TherapyAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Therapy Assistance Prototype")
        self.root.geometry("1150x900")

        self.selected_file = tk.StringVar()
        self._build_ui()

    def _build_ui(self) -> None:
        # Top controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(top_frame, text="Audio file:").pack(side="left")

        self.file_entry = tk.Entry(top_frame, textvariable=self.selected_file, width=80)
        self.file_entry.pack(side="left", padx=5)

        tk.Button(top_frame, text="Browse", command=self.browse_file).pack(side="left", padx=5)
        self.run_button = tk.Button(top_frame, text="Run Pipeline", command=self.run_pipeline_clicked)
        self.run_button.pack(side="left", padx=5)

        self.status_label = tk.Label(self.root, text="Ready", anchor="w")
        self.status_label.pack(fill="x", padx=10)

        # Notebook / Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.prompts_tab = tk.Frame(self.notebook)
        self.results_tab = tk.Frame(self.notebook)

        self.notebook.add(self.prompts_tab, text="Prompts")
        self.notebook.add(self.results_tab, text="Results")

        self._build_prompts_tab()
        self._build_results_tab()

    def _build_prompts_tab(self) -> None:
        # Cleanup Prompt Section
        cleanup_header = tk.Frame(self.prompts_tab)
        cleanup_header.pack(fill="x", padx=10, pady=(10, 0))

        tk.Label(cleanup_header, text="Cleanup Prompt").pack(side="left")
        tk.Button(cleanup_header, text="Load", command=self.load_cleanup_prompt).pack(side="right", padx=2)
        tk.Button(cleanup_header, text="Save", command=self.save_cleanup_prompt).pack(side="right", padx=2)
        tk.Button(cleanup_header, text="Reset", command=self.reset_cleanup_prompt).pack(side="right", padx=2)

        self.cleanup_prompt_text = scrolledtext.ScrolledText(self.prompts_tab, height=14, wrap=tk.WORD)
        self.cleanup_prompt_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.cleanup_prompt_text.insert("1.0", DEFAULT_CLEANUP_PROMPT_TEMPLATE.strip())

        # Summary Prompt Section
        summary_header = tk.Frame(self.prompts_tab)
        summary_header.pack(fill="x", padx=10, pady=(10, 0))

        tk.Label(summary_header, text="Summary Prompt").pack(side="left")
        tk.Button(summary_header, text="Load", command=self.load_summary_prompt).pack(side="right", padx=2)
        tk.Button(summary_header, text="Save", command=self.save_summary_prompt).pack(side="right", padx=2)
        tk.Button(summary_header, text="Reset", command=self.reset_summary_prompt).pack(side="right", padx=2)

        self.summary_prompt_text = scrolledtext.ScrolledText(self.prompts_tab, height=14, wrap=tk.WORD)
        self.summary_prompt_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.summary_prompt_text.insert("1.0", DEFAULT_SUMMARY_PROMPT_TEMPLATE.strip())

    def _build_results_tab(self) -> None:
        tk.Label(self.results_tab, text="Raw Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.raw_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        self.raw_text.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(self.results_tab, text="Cleaned Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.cleaned_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        self.cleaned_text.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Label(self.results_tab, text="Final Summary").pack(anchor="w", padx=10, pady=(10, 0))
        self.summary_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=5)

    def browse_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.selected_file.set(file_path)

    def _load_prompt_into_widget(self, widget: scrolledtext.ScrolledText) -> None:
        file_path = filedialog.askopenfilename(
            title="Load prompt file",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            content = Path(file_path).read_text(encoding="utf-8")
            widget.delete("1.0", tk.END)
            widget.insert("1.0", content)
            self.status_label.config(text=f"Loaded prompt: {file_path}")
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load file:\n{exc}")

    def _save_widget_to_prompt_file(self, widget: scrolledtext.ScrolledText, default_name: str) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Save prompt file",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            content = widget.get("1.0", tk.END).strip()
            Path(file_path).write_text(content, encoding="utf-8")
            self.status_label.config(text=f"Saved prompt: {file_path}")
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save file:\n{exc}")

    def load_cleanup_prompt(self) -> None:
        self._load_prompt_into_widget(self.cleanup_prompt_text)

    def save_cleanup_prompt(self) -> None:
        self._save_widget_to_prompt_file(self.cleanup_prompt_text, "cleanup_prompt.txt")

    def reset_cleanup_prompt(self) -> None:
        self.cleanup_prompt_text.delete("1.0", tk.END)
        self.cleanup_prompt_text.insert("1.0", DEFAULT_CLEANUP_PROMPT_TEMPLATE.strip())
        self.status_label.config(text="Cleanup prompt reset to default")

    def load_summary_prompt(self) -> None:
        self._load_prompt_into_widget(self.summary_prompt_text)

    def save_summary_prompt(self) -> None:
        self._save_widget_to_prompt_file(self.summary_prompt_text, "summary_prompt.txt")

    def reset_summary_prompt(self) -> None:
        self.summary_prompt_text.delete("1.0", tk.END)
        self.summary_prompt_text.insert("1.0", DEFAULT_SUMMARY_PROMPT_TEMPLATE.strip())
        self.status_label.config(text="Summary prompt reset to default")

    def run_pipeline_clicked(self) -> None:
        file_path = self.selected_file.get().strip()
        if not file_path:
            messagebox.showwarning("Missing file", "Please select an audio file first.")
            return

        if not Path(file_path).exists():
            messagebox.showerror("File not found", f"Audio file does not exist:\n{file_path}")
            return

        cleanup_prompt = self.cleanup_prompt_text.get("1.0", tk.END).strip()
        summary_prompt = self.summary_prompt_text.get("1.0", tk.END).strip()

        if "{TRANSCRIPTION}" not in cleanup_prompt:
            messagebox.showerror("Invalid cleanup prompt", "Cleanup prompt must contain {TRANSCRIPTION}")
            return

        if "{TRANSCRIPTION}" not in summary_prompt:
            messagebox.showerror("Invalid summary prompt", "Summary prompt must contain {TRANSCRIPTION}")
            return

        self.run_button.config(state="disabled")
        self.status_label.config(text="Running pipeline... please wait")

        self.raw_text.delete("1.0", tk.END)
        self.cleaned_text.delete("1.0", tk.END)
        self.summary_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._run_pipeline_background,
            args=(file_path, cleanup_prompt, summary_prompt),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_background(self, file_path: str, cleanup_prompt: str, summary_prompt: str) -> None:
        try:
            result = run_pipeline(
                Path(file_path),
                cleanup_prompt_template=cleanup_prompt,
                summary_prompt_template=summary_prompt,
            )
            self.root.after(0, self._update_ui_with_result, result)
        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _update_ui_with_result(self, result: dict) -> None:
        self.raw_text.insert(tk.END, result["raw_transcript"])
        self.cleaned_text.insert(tk.END, result["cleaned_transcript"])
        self.summary_text.insert(tk.END, result["summary"])

        self.status_label.config(
            text=f"Done. Detected language: {result['detected_language']}"
        )
        self.run_button.config(state="normal")

        # Automatically switch to results tab after completion
        self.notebook.select(self.results_tab)

    def _handle_error(self, error_message: str) -> None:
        self.status_label.config(text="Error")
        self.run_button.config(state="normal")
        messagebox.showerror("Pipeline error", error_message)


def main() -> None:
    root = tk.Tk()
    app = TherapyAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
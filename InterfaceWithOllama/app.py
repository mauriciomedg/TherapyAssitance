import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from run_session import (
    run_pipeline,
    run_text_pipeline,
    run_summary_only_pipeline,
    DEFAULT_CLEANUP_PROMPT_TEMPLATE,
    DEFAULT_SUMMARY_PROMPT_TEMPLATE,
)


# =========================
# DARK THEME
# =========================

BG_MAIN = "#1e1e1e"
BG_PANEL = "#252526"
BG_INPUT = "#2d2d30"
BG_BUTTON = "#3c3c3c"
BG_BUTTON_ACTIVE = "#4a4a4a"
FG_MAIN = "#d4d4d4"
FG_MUTED = "#aaaaaa"
ACCENT = "#0e639c"
BORDER = "#3f3f46"
INSERT = "#ffffff"
SELECT_BG = "#264f78"
SELECT_FG = "#ffffff"


def apply_dark_theme(root: tk.Tk) -> None:
    root.configure(bg=BG_MAIN)

    style = ttk.Style()
    style.theme_use("default")

    style.configure(
        "TNotebook",
        background=BG_MAIN,
        borderwidth=0,
    )

    style.configure(
        "TNotebook.Tab",
        background=BG_BUTTON,
        foreground=FG_MAIN,
        padding=(12, 6),
        borderwidth=0,
    )

    style.map(
        "TNotebook.Tab",
        background=[("selected", ACCENT), ("active", BG_BUTTON_ACTIVE)],
        foreground=[("selected", "#ffffff"), ("active", "#ffffff")],
    )


def create_dark_button(parent, text, command):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=BG_BUTTON,
        fg=FG_MAIN,
        activebackground=BG_BUTTON_ACTIVE,
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=10,
        pady=6,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )


def create_dark_label(parent, text, anchor="w"):
    return tk.Label(
        parent,
        text=text,
        bg=BG_MAIN,
        fg=FG_MAIN,
        anchor=anchor,
    )


def create_dark_frame(parent, bg=BG_MAIN):
    return tk.Frame(parent, bg=bg)


def style_text_widget(widget):
    widget.configure(
        bg=BG_INPUT,
        fg=FG_MAIN,
        insertbackground=INSERT,
        selectbackground=SELECT_BG,
        selectforeground=SELECT_FG,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        font=("Consolas", 10),
        padx=8,
        pady=8,
    )


def style_entry_widget(widget):
    widget.configure(
        bg=BG_INPUT,
        fg=FG_MAIN,
        insertbackground=INSERT,
        selectbackground=SELECT_BG,
        selectforeground=SELECT_FG,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )


# =========================
# APP
# =========================

class TherapyAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Therapy Assistance Prototype")
        self.root.geometry("1200x920")
        self.root.configure(bg=BG_MAIN)

        self.selected_file = tk.StringVar()
        self._build_ui()

    def _build_ui(self) -> None:
        # Top controls
        top_frame = create_dark_frame(self.root, BG_MAIN)
        top_frame.pack(fill="x", padx=10, pady=10)

        create_dark_label(top_frame, "Audio file:").pack(side="left")

        self.file_entry = tk.Entry(top_frame, textvariable=self.selected_file, width=55)
        style_entry_widget(self.file_entry)
        self.file_entry.pack(side="left", padx=8)

        create_dark_button(top_frame, "Browse", self.browse_file).pack(side="left", padx=4)

        self.run_full_button = create_dark_button(
            top_frame, "Run Full Pipeline", self.run_pipeline_clicked
        )
        self.run_full_button.pack(side="left", padx=4)

        self.run_text_button = create_dark_button(
            top_frame, "Run From Raw Transcript", self.run_from_text_clicked
        )
        self.run_text_button.pack(side="left", padx=4)

        self.run_summary_only_button = create_dark_button(
            top_frame, "Run Summary Only", self.run_summary_only_clicked
        )
        self.run_summary_only_button.pack(side="left", padx=4)

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            anchor="w",
            bg=BG_MAIN,
            fg=FG_MUTED,
            padx=10,
        )
        self.status_label.pack(fill="x")

        # Notebook / tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.prompts_tab = create_dark_frame(self.notebook, BG_MAIN)
        self.results_tab = create_dark_frame(self.notebook, BG_MAIN)

        self.notebook.add(self.prompts_tab, text="Prompts")
        self.notebook.add(self.results_tab, text="Results")

        self._build_prompts_tab()
        self._build_results_tab()

    def _build_prompts_tab(self) -> None:
        # Cleanup header
        cleanup_header = create_dark_frame(self.prompts_tab, BG_MAIN)
        cleanup_header.pack(fill="x", padx=10, pady=(10, 0))

        create_dark_label(cleanup_header, "Cleanup Prompt").pack(side="left")
        create_dark_button(cleanup_header, "Load", self.load_cleanup_prompt).pack(side="right", padx=2)
        create_dark_button(cleanup_header, "Save", self.save_cleanup_prompt).pack(side="right", padx=2)
        create_dark_button(cleanup_header, "Reset", self.reset_cleanup_prompt).pack(side="right", padx=2)

        self.cleanup_prompt_text = scrolledtext.ScrolledText(self.prompts_tab, height=14, wrap=tk.WORD)
        style_text_widget(self.cleanup_prompt_text)
        self.cleanup_prompt_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.cleanup_prompt_text.insert("1.0", DEFAULT_CLEANUP_PROMPT_TEMPLATE.strip())

        # Summary header
        summary_header = create_dark_frame(self.prompts_tab, BG_MAIN)
        summary_header.pack(fill="x", padx=10, pady=(10, 0))

        create_dark_label(summary_header, "Summary Prompt").pack(side="left")
        create_dark_button(summary_header, "Load", self.load_summary_prompt).pack(side="right", padx=2)
        create_dark_button(summary_header, "Save", self.save_summary_prompt).pack(side="right", padx=2)
        create_dark_button(summary_header, "Reset", self.reset_summary_prompt).pack(side="right", padx=2)

        self.summary_prompt_text = scrolledtext.ScrolledText(self.prompts_tab, height=14, wrap=tk.WORD)
        style_text_widget(self.summary_prompt_text)
        self.summary_prompt_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.summary_prompt_text.insert("1.0", DEFAULT_SUMMARY_PROMPT_TEMPLATE.strip())

    def _build_results_tab(self) -> None:
        create_dark_label(self.results_tab, "Raw Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.raw_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        style_text_widget(self.raw_text)
        self.raw_text.pack(fill="both", expand=True, padx=10, pady=5)

        create_dark_label(self.results_tab, "Cleaned Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.cleaned_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        style_text_widget(self.cleaned_text)
        self.cleaned_text.pack(fill="both", expand=True, padx=10, pady=5)

        create_dark_label(self.results_tab, "Final Summary").pack(anchor="w", padx=10, pady=(10, 0))
        self.summary_text = scrolledtext.ScrolledText(self.results_tab, height=10, wrap=tk.WORD)
        style_text_widget(self.summary_text)
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
            self.status_label.config(text=f"Loaded prompt: {file_path}", fg=FG_MUTED)
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
            self.status_label.config(text=f"Saved prompt: {file_path}", fg=FG_MUTED)
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save file:\n{exc}")

    def load_cleanup_prompt(self) -> None:
        self._load_prompt_into_widget(self.cleanup_prompt_text)

    def save_cleanup_prompt(self) -> None:
        self._save_widget_to_prompt_file(self.cleanup_prompt_text, "cleanup_prompt.txt")

    def reset_cleanup_prompt(self) -> None:
        self.cleanup_prompt_text.delete("1.0", tk.END)
        self.cleanup_prompt_text.insert("1.0", DEFAULT_CLEANUP_PROMPT_TEMPLATE.strip())
        self.status_label.config(text="Cleanup prompt reset to default", fg=FG_MUTED)

    def load_summary_prompt(self) -> None:
        self._load_prompt_into_widget(self.summary_prompt_text)

    def save_summary_prompt(self) -> None:
        self._save_widget_to_prompt_file(self.summary_prompt_text, "summary_prompt.txt")

    def reset_summary_prompt(self) -> None:
        self.summary_prompt_text.delete("1.0", tk.END)
        self.summary_prompt_text.insert("1.0", DEFAULT_SUMMARY_PROMPT_TEMPLATE.strip())
        self.status_label.config(text="Summary prompt reset to default", fg=FG_MUTED)

    def _validate_prompts(self, cleanup_prompt: str, summary_prompt: str, require_cleanup: bool = True) -> bool:
        if require_cleanup and "{TRANSCRIPTION}" not in cleanup_prompt:
            messagebox.showerror("Invalid cleanup prompt", "Cleanup prompt must contain {TRANSCRIPTION}")
            return False

        if "{TRANSCRIPTION}" not in summary_prompt:
            messagebox.showerror("Invalid summary prompt", "Summary prompt must contain {TRANSCRIPTION}")
            return False

        return True

    def _disable_run_buttons(self) -> None:
        self.run_full_button.config(state="disabled")
        self.run_text_button.config(state="disabled")
        self.run_summary_only_button.config(state="disabled")

    def _enable_run_buttons(self) -> None:
        self.run_full_button.config(state="normal")
        self.run_text_button.config(state="normal")
        self.run_summary_only_button.config(state="normal")

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

        if not self._validate_prompts(cleanup_prompt, summary_prompt, require_cleanup=True):
            return

        self._disable_run_buttons()
        self.status_label.config(text="Running full pipeline... please wait", fg=FG_MUTED)

        self.raw_text.delete("1.0", tk.END)
        self.cleaned_text.delete("1.0", tk.END)
        self.summary_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._run_pipeline_background,
            args=(file_path, cleanup_prompt, summary_prompt),
            daemon=True,
        )
        thread.start()

    def run_from_text_clicked(self) -> None:
        raw_transcript = self.raw_text.get("1.0", tk.END).strip()
        if not raw_transcript:
            messagebox.showwarning("Missing transcript", "Raw Transcript is empty.")
            return

        cleanup_prompt = self.cleanup_prompt_text.get("1.0", tk.END).strip()
        summary_prompt = self.summary_prompt_text.get("1.0", tk.END).strip()

        if not self._validate_prompts(cleanup_prompt, summary_prompt, require_cleanup=True):
            return

        self._disable_run_buttons()
        self.status_label.config(text="Running cleanup + summary from raw transcript...", fg=FG_MUTED)

        self.cleaned_text.delete("1.0", tk.END)
        self.summary_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._run_text_pipeline_background,
            args=(raw_transcript, cleanup_prompt, summary_prompt),
            daemon=True,
        )
        thread.start()

    def run_summary_only_clicked(self) -> None:
        cleaned_transcript = self.cleaned_text.get("1.0", tk.END).strip()
        if not cleaned_transcript:
            messagebox.showwarning("Missing cleaned transcript", "Cleaned Transcript is empty.")
            return

        summary_prompt = self.summary_prompt_text.get("1.0", tk.END).strip()

        if "{TRANSCRIPTION}" not in summary_prompt:
            messagebox.showerror("Invalid summary prompt", "Summary prompt must contain {TRANSCRIPTION}")
            return

        self._disable_run_buttons()
        self.status_label.config(text="Running summary only from cleaned transcript...", fg=FG_MUTED)

        self.summary_text.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self._run_summary_only_background,
            args=(cleaned_transcript, summary_prompt),
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

    def _run_text_pipeline_background(self, raw_transcript: str, cleanup_prompt: str, summary_prompt: str) -> None:
        try:
            result = run_text_pipeline(
                raw_transcript,
                cleanup_prompt_template=cleanup_prompt,
                summary_prompt_template=summary_prompt,
            )
            self.root.after(0, self._update_ui_with_result, result)
        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _run_summary_only_background(self, cleaned_transcript: str, summary_prompt: str) -> None:
        try:
            result = run_summary_only_pipeline(
                cleaned_transcript,
                summary_prompt_template=summary_prompt,
            )
            self.root.after(0, self._update_ui_with_result, result)
        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))

    def _update_ui_with_result(self, result: dict) -> None:
        if result.get("raw_transcript"):
            self.raw_text.delete("1.0", tk.END)
            self.raw_text.insert(tk.END, result["raw_transcript"])

        if result.get("cleaned_transcript"):
            self.cleaned_text.delete("1.0", tk.END)
            self.cleaned_text.insert(tk.END, result["cleaned_transcript"])

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, result["summary"])

        self.status_label.config(
            text=f"Done. Detected language: {result.get('detected_language', 'unknown')}",
            fg=FG_MUTED,
        )
        self._enable_run_buttons()
        self.notebook.select(self.results_tab)

    def _handle_error(self, error_message: str) -> None:
        self.status_label.config(text="Error", fg="#ff6b6b")
        self._enable_run_buttons()
        messagebox.showerror("Pipeline error", error_message)


def main() -> None:
    root = tk.Tk()
    apply_dark_theme(root)
    app = TherapyAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
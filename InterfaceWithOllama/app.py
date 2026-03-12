import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from run_session import run_pipeline


class TherapyAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Therapy Assistance Prototype")
        self.root.geometry("1000x780")

        self.selected_file = tk.StringVar()

        self._build_ui()

    def _build_ui(self) -> None:
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

        tk.Label(self.root, text="Raw Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.raw_text = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.raw_text.pack(fill="both", expand=False, padx=10, pady=5)

        tk.Label(self.root, text="Cleaned Transcript").pack(anchor="w", padx=10, pady=(10, 0))
        self.cleaned_text = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.cleaned_text.pack(fill="both", expand=False, padx=10, pady=5)

        tk.Label(self.root, text="Final Summary").pack(anchor="w", padx=10, pady=(10, 0))
        self.summary_text = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
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

    def run_pipeline_clicked(self) -> None:
        file_path = self.selected_file.get().strip()
        if not file_path:
            messagebox.showwarning("Missing file", "Please select an audio file first.")
            return

        if not Path(file_path).exists():
            messagebox.showerror("File not found", f"Audio file does not exist:\n{file_path}")
            return

        self.run_button.config(state="disabled")
        self.status_label.config(text="Running pipeline... please wait")

        self.raw_text.delete("1.0", tk.END)
        self.cleaned_text.delete("1.0", tk.END)
        self.summary_text.delete("1.0", tk.END)

        thread = threading.Thread(target=self._run_pipeline_background, args=(file_path,), daemon=True)
        thread.start()

    def _run_pipeline_background(self, file_path: str) -> None:
        try:
            result = run_pipeline(Path(file_path))
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
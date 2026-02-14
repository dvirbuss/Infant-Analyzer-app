# ui/gui_ui.py
from __future__ import annotations

import datetime as dt
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
from tkcalendar import DateEntry
from tkinterdnd2 import DND_FILES
import cv2
from PIL import Image, ImageTk

import config
from core.pipeline import run as run_pipeline

POSES = ["Prone", "Supine", "Sitting"]


def parse_drop_path(data: str) -> str:
    p = data.strip()
    if p.startswith("{") and p.endswith("}"):
        p = p[1:-1]
    return p


class InfantAnalyzerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Infant Motor Development Analyzer")
        self.root.configure(bg="#f0f7ff")
        self.root.resizable(False, False)

        # center window
        w, h = 650, 720
        sx, sy = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = int((sx / 2) - (w / 2))
        y = int((sy / 2) - (h / 2))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # state
        self.pose_videos = {p: {"path": None, "loaded": False} for p in POSES}
        self.selected_pose = tk.StringVar(value="Prone")
        self.processing = False
        self.cancel_event = threading.Event()

        # birthdate state (default = 50 days ago)
        self.birthday_var = tk.StringVar()
        default_date = dt.date.today() - dt.timedelta(days=60)
        self.birthday_var.set(default_date.strftime("%d/%m/%y"))

        self._build_ui()
        self.init_preview_window()
        self.update_generate_state()

    # ---------------- Preview ----------------

    def init_preview_window(self):
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Live Preview")
        self.preview_window.geometry("660x560")
        self.preview_window.configure(bg="black")

        self.preview_label = tk.Label(self.preview_window, bg="black")
        self.preview_label.pack(fill="both", expand=True)

        # Optional: a small status line under the preview
        self.preview_status = tk.Label(
            self.preview_window, text="", bg="black", fg="white", font=("Segoe UI", 10)
        )
        self.preview_status.pack(fill="x")

        self.preview_window.withdraw()
        self.preview_window.protocol("WM_DELETE_WINDOW", self.preview_window.withdraw)

    def show_preview_window(self):
        if hasattr(self, "preview_window") and self.preview_window.winfo_exists():
            self.preview_window.deiconify()
            self.preview_window.lift()

    def update_preview_frame(self, frame):
        if not hasattr(self, "preview_label"):
            return
        if not self.preview_window.winfo_exists():
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)

        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk)

    def set_preview_status(self, text: str):
        if hasattr(self, "preview_status"):
            self.preview_status.config(text=text)

    # ---------------- UI ----------------

    def _build_ui(self):
        tk.Label(
            self.root,
            text="Infant Motor Development Analyzer",
            font=("Helvetica", 20, "bold"),
            bg="#f0f7ff",
            fg="#003366",
        ).pack(pady=(15, 10))

        # birthday
        tk.Label(
            self.root,
            text="Choose Infant Birthday:",
            font=("Segoe UI", 15),
            bg="#f0f7ff",
            fg="#002244",
        ).pack(pady=(10, 5))

        self.calendar = DateEntry(
            self.root,
            textvariable=self.birthday_var,
            width=14,
            background="#0066cc",
            foreground="white",
            borderwidth=0,
            font=("Segoe UI", 11),
            date_pattern="dd/mm/yy",
            selectbackground="#3399ff",
            selectforeground="white",
        )
        default_date = dt.date.today() - dt.timedelta(days=61)
        self.calendar.set_date(default_date)

        self.calendar.pack(pady=5)
        self.calendar.bind("<<DateEntrySelected>>", lambda e: self.update_generate_state())

        self.age_status_label = tk.Label(self.root, text="", font=("Segoe UI", 10), bg="#f0f7ff")
        self.age_status_label.pack(pady=(0, 10))

        # pose selector
        pose_row = tk.Frame(self.root, bg="#f0f7ff")
        pose_row.pack(pady=(5, 10))

        tk.Label(
            pose_row,
            text="Run pose:",
            font=("Segoe UI", 11, "bold"),
            bg="#f0f7ff",
            fg="#003366",
        ).pack(side="left", padx=(0, 8))

        for p in POSES:
            tk.Radiobutton(
                pose_row,
                text=p,
                variable=self.selected_pose,
                value=p,
                bg="#f0f7ff",
                fg="#003366",
                activebackground="#f0f7ff",
                command=self.update_generate_state,
            ).pack(side="left", padx=6)

        # video inputs
        tk.Label(
            self.root,
            text="Load Infant Videos:",
            font=("Segoe UI", 15),
            bg="#f0f7ff",
            fg="#002244",
        ).pack(pady=(15, 5))

        self.drop_frame = tk.Frame(self.root, bg="#f0f7ff")
        self.drop_frame.pack(pady=10)

        self._create_video_input("Prone", 0, config.PRONE_ICON)
        self._create_video_input("Supine", 1, config.SUPINE_ICON)
        self._create_video_input("Sitting", 2, config.SITTING_ICON)

        self.status_label = tk.Label(self.root, text="", font=("Segoe UI", 10), bg="#f0f7ff", fg="#003366")
        self.status_label.pack(pady=(10, 5))

        self.bottom_bar = tk.Frame(self.root, bg="#f0f7ff")
        self.bottom_bar.pack(side="bottom", fill="x", pady=15)

        # Create buttons INSIDE the bottom bar (important!)
        self.generate_button = tk.Button(
            self.bottom_bar,
            text="Generate",
            command=self.on_generate_clicked,
            font=("Segoe UI", 11, "bold"),
            bg="lightgray",
            fg="gray",
            activebackground="#218838",
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            state=tk.DISABLED,
            cursor="arrow",
        )

        self.stop_button = tk.Button(
            self.bottom_bar,
            text="Stop",
            command=self.on_stop_clicked,
            font=("Segoe UI", 11, "bold"),
            bg="#dc3545",
            fg="white",
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            state=tk.DISABLED,
        )

        # Put them next to each other
        self.bottom_bar.pack(side="bottom", pady=15)
        self.generate_button.pack(side="left", padx=(25, 10))
        self.stop_button.pack(side="left", padx=10)
        self.bottom_bar.pack_configure(anchor="center")

    def _create_video_input(self, pose: str, col: int, icon_path: Path):
        tk.Label(
            self.drop_frame,
            text=pose,
            font=("Segoe UI", 11, "bold"),
            bg="#f0f7ff",
            fg="#003366",
        ).grid(row=0, column=col, pady=(0, 5))

        if icon_path and icon_path.exists():
            img = tk.PhotoImage(file=str(icon_path)).subsample(2, 2)
            lbl_img = tk.Label(self.drop_frame, image=img, bg="#f0f7ff")
            lbl_img.image = img
            lbl_img.grid(row=1, column=col, pady=(0, 5))

        tk.Button(
            self.drop_frame,
            text=f"Open {pose} Video",
            command=lambda p=pose: self.open_video(p),
            font=("Segoe UI", 9),
            bg="white",
            fg="#003366",
        ).grid(row=2, column=col, pady=(0, 5))

        drop_area = tk.Label(
            self.drop_frame,
            text="Drop Video Here",
            width=18,
            height=5,
            bg="#d0e8ff",
            relief="ridge",
            bd=2,
            font=("Segoe UI", 9, "bold"),
        )
        drop_area.grid(row=3, column=col, pady=(0, 5), padx=15)
        drop_area.drop_target_register(DND_FILES)
        drop_area.dnd_bind("<<Drop>>", lambda e, p=pose: self.on_drop(e, p))

        success_lbl = tk.Label(self.drop_frame, text="", font=("Segoe UI", 9), bg="#f0f7ff", fg="green")
        success_lbl.grid(row=4, column=col)
        success_lbl.grid_remove()

        self.pose_videos[pose]["success_lbl"] = success_lbl

        show_btn = tk.Button(
            self.drop_frame,
            text="Show Video",
            command=lambda p=pose: self.show_video(p),
            font=("Segoe UI", 9),
            bg="#e6f2ff",
            fg="black",
            relief="groove",
            bd=1,
        )
        show_btn.grid(row=5, column=col, pady=(3, 0))
        show_btn.grid_remove()

        self.pose_videos[pose]["show_btn"] = show_btn

    def show_video(self, pose: str):
        path = self.pose_videos.get(pose, {}).get("path")
        if not path:
            messagebox.showerror("Error", f"No video loaded for {pose}.")
            return

        def _play():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", f"Cannot open video:\n{path}"))
                return

            win = f"{pose} Video"
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(win, frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break

            cap.release()
            cv2.destroyWindow(win)

        threading.Thread(target=_play, daemon=True).start()

    # ---------------- logic ----------------
    def on_stop_clicked(self):
        if self.processing:
            self.cancel_event.set()
            self.status_label.config(text="Stopping… saving partial output.")

    def get_birthdate(self) -> dt.date | None:
        try:
            return dt.datetime.strptime(self.birthday_var.get(), "%d/%m/%y").date()
        except Exception:
            return None

    def is_age_valid(self) -> bool:
        bd = self.get_birthdate()
        if bd is None:
            self.age_status_label.config(text="Invalid date.", fg="red")
            return False

        age_days = (dt.date.today() - bd).days
        age_months = round(age_days / 30.44, 2)

        if age_days < 0:
            self.age_status_label.config(text=f"Biological Age: {age_months} months\nAge must be positive", fg="red")
            return False
        if age_days < 60:
            self.age_status_label.config(text=f"Biological Age: {age_months} months\nToo young (<2 months)", fg="red")
            return False
        if age_days > 180:
            self.age_status_label.config(text=f"Biological Age: {age_months} months\nToo old (>6 months)", fg="red")
            return False

        self.age_status_label.config(text=f"Biological Age: {age_months} months\nValid age", fg="green")
        return True

    def any_video_loaded(self) -> bool:
        return any(v["loaded"] for v in self.pose_videos.values())

    def update_generate_state(self):
        age_ok = self.is_age_valid()
        any_video_ok = self.any_video_loaded()
        enabled = (not self.processing) and age_ok and any_video_ok

        if enabled:
            self.generate_button.config(state=tk.NORMAL, bg="#28a745", fg="white", cursor="hand2")
        else:
            self.generate_button.config(state=tk.DISABLED, bg="lightgray", fg="gray", cursor="arrow")

    def open_video(self, pose: str):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self._set_pose_video(pose, file_path)

    def on_drop(self, event, pose: str):
        path = parse_drop_path(event.data)
        if Path(path).exists():
            self._set_pose_video(pose, path)

    def _set_pose_video(self, pose: str, path: str):
        self.pose_videos[pose]["path"] = path
        self.pose_videos[pose]["loaded"] = True

        lbl = self.pose_videos[pose].get("success_lbl")
        if lbl:
            lbl.config(text="Video loaded ✅")
            lbl.grid()

        self.status_label.config(text=f"{pose} video set: {Path(path).name}")
        self.update_generate_state()
        btn = self.pose_videos[pose].get("show_btn")
        if btn:
            btn.grid()

    def on_generate_clicked(self):
        if self.processing:
            return

        pose = self.selected_pose.get()
        video_path = self.pose_videos.get(pose, {}).get("path")

        if not video_path:
            messagebox.showerror("Error", f"No video loaded for {pose}.")
            return

        birthdate = self.get_birthdate()
        if birthdate is None:
            messagebox.showerror("Error", "Invalid birthdate.")
            return

        self.processing = True
        self.status_label.config(text="Processing... (this may take a while)")
        self.update_generate_state()

        self.root.after(0, self.show_preview_window)
        self.cancel_event.clear()
        self.stop_button.config(state=tk.NORMAL)

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(pose, video_path, birthdate),
            daemon=True,
        )

        thread.start()

    def _run_pipeline_thread(self, pose: str, video_path: str, birthdate: dt.date):
        try:
            video_name = Path(video_path).stem
            time_stamp = dt.datetime.now().strftime("%d-%m-%y_%H-%M")
            folder_name = f"{pose.lower()}_{video_name}_{time_stamp}"
            out_dir = Path(config.VIDEOS_OUTPUT_DIR) / folder_name

            self.root.after(0, self.show_preview_window)

            progress_callback = lambda frame_idx, fps, stopping: self.root.after(
                0,
                lambda: self.set_preview_status(
                    f"{'Stopping… ' if stopping else ''}Frame: {frame_idx} | Time: {frame_idx / fps:.2f}s") )

            result = run_pipeline(
                pose=pose,
                video_path=video_path,
                birthdate=birthdate,
                runner="gui",
                frame_callback=lambda frame: self.root.after(0, lambda f=frame: self.update_preview_frame(f)),
                cancel_check=self.cancel_event.is_set,
                progress_callback=progress_callback
            )

            if result.get("cancelled"):
                frames = result.get("artifacts", {}).get("frames_processed", "?")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Stopped",
                        f"Stopped early ✅\nSaved partial output.\n\nFrames processed: {frames}\nFolder:\n{out_dir}"
                    ),
                )
                self.root.after(0, lambda: self.status_label.config(text=f"Stopped. Partial outputs: {out_dir}"))
                return

            msg = (
                f"Done ✅\n\n"
                f"Pose: {result['pose']}\n"
                f"Age (months): {result['age_months']}\n"
                f"AIMS score: {result['aims_score']}\n\n"
                f"Outputs folder:\n{out_dir}"
            )
            self.root.after(0, lambda: messagebox.showinfo("Success", msg))
            self.root.after(0, lambda: self.status_label.config(text=f"Finished. Outputs: {out_dir}"))

        except Exception:
            tb = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", tb))
            self.root.after(0, lambda: self.status_label.config(text="Error. See message box."))

        finally:
            self.processing = False
            self.stop_button.config(state=tk.DISABLED)
            self.root.after(0, self.update_generate_state)
            self.cancel_event.clear()

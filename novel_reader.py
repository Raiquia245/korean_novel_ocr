"""
Korean Web Novel OCR + Translator
==================================

Reads on-screen Korean text (e.g. a web-novel reader that blocks copy/paste)
and shows an instant English translation — no manual screenshots, no copy/paste.

Workflow:
  1. On first run you select the screen region where the novel text appears
     (drag a box around it). The region is remembered for the session.
  2. Press the hotkey (default F9) whenever you want to read the current page:
     the tool captures that region, runs Korean OCR, translates the text, and
     shows both the original and the translation in an always-on-top window.
  3. Press ESC to quit.

Engines:
  - OCR:        EasyOCR (Korean model, downloaded once on first use)
  - Translate:  Google Translate (free) or Gemini AI (needs a free API key)
  - Capture:    mss (fast screen capture)
  - Overlay:    Tkinter always-on-top window

Note: intended for personal reading of content you already have legitimate
access to. Do not use it to scrape or redistribute content.
"""

import os
import queue
import threading
import time

import numpy as np
import mss
from deep_translator import GoogleTranslator
from pynput import keyboard

# Load a .env file if present, so GEMINI_API_KEY can live there instead of the
# shell. Optional — falls back to real environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# EasyOCR is heavy to import, so we do it once at startup and reuse the reader.
import easyocr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOTKEY = keyboard.Key.f9          # press to capture + translate
RESET_KEY = keyboard.Key.f10      # press to start a new novel/chapter (clears context)
QUIT_KEY = keyboard.Key.esc       # press to exit
SOURCE_LANG = "ko"                # OCR + translate source
TARGET_LANG = "en"                # translation target (English)

# Which translation engine to use: "google" (free, no key) or "gemini" (AI).
TRANSLATE_ENGINE = "google"

# Show results in an always-on-top overlay window (True) or just the console.
USE_OVERLAY = True

# Gemini settings (only used when TRANSLATE_ENGINE == "gemini").
# The API key is read from the GEMINI_API_KEY environment variable (or a .env
# file) — never hardcode it. Get a free key at https://aistudio.google.com/
# Model names change fairly often; verify the current free model ID in Google
# AI Studio and update this if needed.
GEMINI_MODEL = "gemini-2.5-flash"

# When True, the last few pages are sent to Gemini as context so character
# names and terms stay consistent across a chapter.
GEMINI_USE_CONTEXT = True
GEMINI_CONTEXT_PAGES = 3  # how many recent pages to keep as context

GEMINI_SYSTEM_PROMPT = (
    "You are a professional literary translator for Korean web novels. "
    "Translate the given Korean text into natural, fluent English prose that "
    "preserves the tone, dialogue, and narrative voice of a novel. Keep "
    "character names and Korean honorifics consistent with any previous pages "
    "provided as context. Return ONLY the English translation, with no notes "
    "or explanations."
)


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------
def select_region():
    """
    Let the user drag a rectangle over the screen to choose the capture area.
    Returns a dict {top, left, width, height} suitable for mss, or None.
    """
    import cv2

    title = "Drag to select the text area, then press ENTER. ESC to cancel."

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        shot = sct.grab(monitor)
        img = np.array(shot)[:, :, :3]

    clone = img.copy()
    box = {}
    drawing = {"active": False, "x0": 0, "y0": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing.update(active=True, x0=x, y0=y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            disp = clone.copy()
            cv2.rectangle(disp, (drawing["x0"], drawing["y0"]), (x, y), (0, 255, 0), 2)
            cv2.imshow(title, disp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["active"] = False
            box.update(
                left=min(drawing["x0"], x) + monitor["left"],
                top=min(drawing["y0"], y) + monitor["top"],
                width=abs(x - drawing["x0"]),
                height=abs(y - drawing["y0"]),
            )
            disp = clone.copy()
            cv2.rectangle(disp, (drawing["x0"], drawing["y0"]), (x, y), (0, 255, 0), 2)
            cv2.imshow(title, disp)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and box.get("width"):   # ENTER
            break
        if key == 27:                          # ESC
            box.clear()
            break
    cv2.destroyAllWindows()

    return box if box.get("width") else None


# ---------------------------------------------------------------------------
# Capture + OCR
# ---------------------------------------------------------------------------
def capture_region(region):
    """Grab the chosen screen region and return it as a BGR numpy array."""
    with mss.mss() as sct:
        shot = sct.grab(region)
    return np.array(shot)[:, :, :3]


def ocr_korean(reader, image):
    """Run EasyOCR on the image and return the recognized Korean text."""
    results = reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(results).strip()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
# Rolling history of recent pages, used as context for Gemini.
_page_history = []


def reset_context():
    """
    Clear the page history — call this when switching to a different novel or
    chapter so names/terms from the previous book don't leak into the new one.
    """
    _page_history.clear()


def translate_text(text, target=TARGET_LANG):
    """Translate Korean text using the configured engine."""
    if not text:
        return ""
    if TRANSLATE_ENGINE == "gemini":
        return _translate_gemini(text)
    return _translate_google(text, target)


def _translate_google(text, target):
    """Free Google Translate backend (deep-translator), no API key."""
    try:
        return GoogleTranslator(source="ko", target=target).translate(text)
    except Exception as e:
        return f"[translation error: {e}]"


# The Gemini client is created once and reused across captures.
_gemini_model = None


def _get_gemini_model():
    """
    Lazily create and cache the Gemini model client.

    Reads GEMINI_API_KEY from the environment. Raises a clear error if the key
    or the library is missing, so the user knows exactly what to fix.
    """
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/ and store it in .env or export it."
        )

    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "The 'google-generativeai' package is not installed. "
            "Run: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(
        GEMINI_MODEL, system_instruction=GEMINI_SYSTEM_PROMPT
    )
    return _gemini_model


def _translate_gemini(text):
    """
    Translate via the Gemini API. Optionally includes recent pages as context
    for consistency, and retries on 429 rate-limit errors with backoff.
    """
    try:
        model = _get_gemini_model()
    except RuntimeError as e:
        return f"[gemini setup error: {e}]"

    if GEMINI_USE_CONTEXT and _page_history:
        context = "\n\n".join(_page_history[-GEMINI_CONTEXT_PAGES:])
        prompt = (
            "Previous pages (for name/term consistency only, do not re-translate):\n"
            f"{context}\n\n"
            "Now translate this new page:\n"
            f"{text}"
        )
    else:
        prompt = text

    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            out = (resp.text or "").strip()
            if GEMINI_USE_CONTEXT:
                _page_history.append(text)
                del _page_history[:-GEMINI_CONTEXT_PAGES]  # keep it bounded
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            return f"[gemini error: {e}]"
    return "[gemini error: rate limited — please try again shortly]"


# ---------------------------------------------------------------------------
# Always-on-top overlay window (Tkinter)
# ---------------------------------------------------------------------------
class Overlay:
    """
    A small always-on-top window that shows the latest translation.

    Tkinter must run on the main thread, so the overlay owns the main loop and
    receives updates from the worker thread through a thread-safe queue.
    """

    def __init__(self, update_queue):
        import tkinter as tk
        self.tk = tk
        self.q = update_queue

        self.root = tk.Tk()
        self.root.title("Novel Translation")
        self.root.attributes("-topmost", True)
        self.root.geometry("460x620+40+40")
        self.root.configure(bg="#1e1e1e")

        header = tk.Label(
            self.root, text="F9 = read page  •  F10 = new novel  •  ESC = quit",
            bg="#1e1e1e", fg="#888888", font=("Segoe UI", 9),
        )
        header.pack(fill="x", padx=10, pady=(8, 4))

        self.status = tk.Label(
            self.root, text="Ready.", bg="#1e1e1e", fg="#4ec9b0",
            font=("Segoe UI", 9, "italic"), anchor="w",
        )
        self.status.pack(fill="x", padx=10)

        # Scrollable text area for the translation.
        wrap = tk.Frame(self.root, bg="#1e1e1e")
        wrap.pack(fill="both", expand=True, padx=10, pady=8)
        scrollbar = tk.Scrollbar(wrap)
        scrollbar.pack(side="right", fill="y")
        self.text = tk.Text(
            wrap, wrap="word", bg="#252526", fg="#e0e0e0",
            font=("Georgia", 12), padx=10, pady=10, relief="flat",
            yscrollcommand=scrollbar.set,
        )
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text.yview)

        self.text.tag_configure("ko", foreground="#9cdcfe", font=("Malgun Gothic", 10))
        self.text.tag_configure("en", foreground="#e0e0e0", font=("Georgia", 12))

        self.root.after(100, self._poll_queue)

    def set_status(self, msg, color="#4ec9b0"):
        self.status.config(text=msg, fg=color)

    def show(self, korean, english):
        self.text.delete("1.0", "end")
        self.text.insert("end", english + "\n\n", "en")
        self.text.insert("end", "— original —\n", "ko")
        self.text.insert("end", korean, "ko")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.set_status(*payload) if isinstance(payload, tuple) else self.set_status(payload)
                elif kind == "result":
                    korean, english = payload
                    self.show(korean, english)
                    self.set_status("Done.", "#4ec9b0")
                elif kind == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Worker: capture -> OCR -> translate
# ---------------------------------------------------------------------------
def worker_loop(reader, region, state, ui_queue):
    """Run in a background thread; pushes updates to the overlay/console."""
    while not state["quit"]:
        if state.get("reset"):
            state["reset"] = False
            reset_context()
            _emit(ui_queue, "status", ("New novel/chapter — context cleared.", "#4ec9b0"))
            print("[*] Context cleared for a new novel/chapter.\n")
        if state["capture"]:
            state["capture"] = False
            _emit(ui_queue, "status", ("Capturing & reading…", "#dcdcaa"))
            img = capture_region(region)
            korean = ocr_korean(reader, img)
            if not korean:
                _emit(ui_queue, "status", ("No text detected — adjust the area.", "#f48771"))
                print("[!] No text detected. Try adjusting the selected area.\n")
                continue
            _emit(ui_queue, "status", ("Translating…", "#dcdcaa"))
            english = translate_text(korean)

            # Console output (always).
            print("-" * 60)
            print("[KO]\n" + korean)
            print("\n[EN]\n" + english)
            print("-" * 60 + "\n")

            _emit(ui_queue, "result", (korean, english))
        time.sleep(0.05)

    _emit(ui_queue, "quit", None)


def _emit(ui_queue, kind, payload):
    if ui_queue is not None:
        ui_queue.put((kind, payload))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Korean Web Novel OCR + Translator")
    print("=" * 60)
    engine_desc = (f"{GEMINI_MODEL}" if TRANSLATE_ENGINE == "gemini"
                   else "Google Translate (free)")
    print(f"Translation engine: {TRANSLATE_ENGINE} ({engine_desc})")
    print(f"Target language: {TARGET_LANG.upper()}")
    if TRANSLATE_ENGINE == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("[!] Warning: GEMINI_API_KEY is not set — translation will fail.")

    print("Loading EasyOCR model (Korean)… (slow the first time)")
    reader = easyocr.Reader([SOURCE_LANG], gpu=False)
    print("Model ready.\n")

    print("Step 1: select the novel text area (drag a box, then press ENTER).")
    region = select_region()
    if not region:
        print("Cancelled.")
        return
    print(f"Area selected: {region}\n")
    print("Step 2: press [F9] to read a page, [F10] when starting a new novel/chapter, [ESC] to quit.\n")

    state = {"capture": False, "quit": False, "reset": False}

    def on_press(key):
        if key == HOTKEY:
            state["capture"] = True
        elif key == RESET_KEY:
            state["reset"] = True
        elif key == QUIT_KEY:
            state["quit"] = True
            return False  # stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    if USE_OVERLAY:
        # Overlay owns the main thread; worker runs in the background.
        ui_queue = queue.Queue()
        try:
            overlay = Overlay(ui_queue)
        except Exception as e:
            print(f"[!] Could not start overlay ({e}); falling back to console.")
            _run_console(reader, region, state)
            listener.stop()
            return

        t = threading.Thread(
            target=worker_loop, args=(reader, region, state, ui_queue), daemon=True
        )
        t.start()
        overlay.run()          # blocks until the window closes
        state["quit"] = True   # ensure worker stops
    else:
        _run_console(reader, region, state)

    listener.stop()
    print("Exited. Happy reading!")


def _run_console(reader, region, state):
    """Console-only loop (no overlay)."""
    while not state["quit"]:
        if state.get("reset"):
            state["reset"] = False
            reset_context()
            print("[*] Context cleared for a new novel/chapter.\n")
        if state["capture"]:
            state["capture"] = False
            img = capture_region(region)
            korean = ocr_korean(reader, img)
            if not korean:
                print("[!] No text detected. Try adjusting the selected area.\n")
                continue
            english = translate_text(korean)
            print("-" * 60)
            print("[KO]\n" + korean)
            print("\n[EN]\n" + english)
            print("-" * 60 + "\n")
        time.sleep(0.05)


if __name__ == "__main__":
    main()

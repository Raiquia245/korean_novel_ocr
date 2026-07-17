# Korean Web Novel OCR + Translator

A desktop reading aid for Korean web novels shown in apps that block copy/paste. Instead of manually screenshotting each page, you select the text area once, then press a hotkey to instantly OCR and translate whatever is on screen into English — shown in an always-on-top overlay window.

Built to read Korean web novels (e.g. a Windows reader that disables text selection) for personal use.

## How it works

1. **Select region once** — on launch, drag a box over the area where the novel text appears.
2. **Press F9 to read** — the tool captures that region, runs Korean OCR, translates it to English, and shows both the original and the translation in a floating window.
3. **Press F10 when you switch to a different novel or chapter** — this clears the AI's page context so names/terms from the previous book don't leak into the new one.
4. **Press ESC to quit.**

## Engines

| Job | Library | Notes |
|-----|---------|-------|
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Korean model, downloaded once on first run. More accurate on thin on-screen Korean fonts than Tesseract. |
| Translation | [deep-translator](https://github.com/nidhaloff/deep-translator) *or* Gemini API | Google Translate (free, no key) **or** an AI model for more natural novel prose. |
| Screen capture | [mss](https://github.com/BoboTiG/python-mss) | Fast, cross-platform region capture. |
| Hotkey | [pynput](https://github.com/moses-palmer/pynput) | Global F9 / ESC listener. |
| Overlay | Tkinter (stdlib) | Always-on-top translation window. |

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the EasyOCR Korean model (a few hundred MB), which is cached afterward.

## Usage

```bash
python novel_reader.py
```

Then: drag to select the text area -> press **ENTER** to confirm -> press **F9** whenever you want the current page read -> **F10** when starting a different novel/chapter -> **ESC** to exit.

### Settings (top of `novel_reader.py`)

- `TARGET_LANG = "en"` — translation target (English by default).
- `USE_OVERLAY = True` — show the floating window; set `False` for console-only.
- `TRANSLATE_ENGINE = "google"` — `"google"` (free) or `"gemini"` (AI).

## Translation engines

- **`"google"`** (default) — free, no API key, uses Google Translate. Fast and fine for gist reading.
- **`"gemini"`** — sends the OCR'd text to Google's Gemini AI, which produces more natural, novel-appropriate prose because it follows a translation instruction (preserve tone, dialogue, honorifics). It can also keep the last few pages as context so character names and terms stay consistent across a chapter. Requires a free API key.

### Using Gemini

1. Get a free API key at <https://aistudio.google.com/> (no credit card needed).
2. Copy `.env.example` to `.env` and paste your key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Install the extra dependency: `pip install google-generativeai python-dotenv`
4. In `novel_reader.py`, set `TRANSLATE_ENGINE = "gemini"`.

Notes on the Gemini free tier (verify current values in Google AI Studio, as they change):
- Free models are currently the **Flash / Flash-Lite** family; Pro models became paid-only in 2026. Update `GEMINI_MODEL` if the model ID changes.
- Rate limits are generous for one-page-per-keypress reading (roughly 15 requests/min, ~1,500/day for Flash). The code retries automatically on rate-limit (429) errors.
- Chapter context is controlled by `GEMINI_USE_CONTEXT` and `GEMINI_CONTEXT_PAGES`. The tool doesn't know when you switch books, so **press F10 to clear the context** whenever you start a different novel or chapter — otherwise names/terms from the previous book can carry over.
- **Privacy:** on the free tier, Google may use your inputs to improve its models. Fine for public novel text; don't send anything sensitive.

## Alternative: batch OCR from image files

If you already have saved screenshots, `batch_ocr.py` (Tesseract-based, with optional preprocessing) and `batch_ocr_simple.py` process a whole folder of images into a text file. These are the earlier versions of this project, kept for when you want file-based batch extraction rather than live screen reading.

```bash
python batch_ocr.py          # interactive: single image / layout / batch folder
python batch_ocr_simple.py   # scans the ./extract folder
```

## Notes & limitations

- **OCR accuracy** depends on font size and contrast. If results are poor, select a tighter region around just the text, or increase the reader's font size.
- **First run is slow** while the OCR model downloads and loads; subsequent captures are fast.
- **GPU:** the reader runs on CPU by default (`gpu=False`). If you have a CUDA GPU, set `gpu=True` in `easyocr.Reader(...)` for faster OCR.
- **Fair use:** intended for personal reading of content you already have legitimate access to. Do not use it to scrape, republish, or redistribute content.

## Possible future improvements

- Auto-capture mode that re-reads every N seconds for continuous scrolling.
- Cache/deduplicate identical captures to avoid re-translating the same page.
- Save a running English transcript of the chapter to a file.

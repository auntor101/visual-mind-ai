---
title: VisualMind AI
emoji: 👁️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# VisualMind AI

Upload any image and instantly get an AI-generated caption, a detailed analysis, and a chat interface to ask follow-up questions about it.

## What it does

1. **Upload an image** (JPEG, PNG, WebP, or GIF up to 4 MB)
2. **BLIP** (HuggingFace) generates a quick caption
3. **Llama 4 Scout** (Groq) gives a deep analysis — objects, colors, text, context
4. **Ask follow-up questions** in the chat — the AI remembers the image and your conversation

## What you need

Two free API keys — no credit card required:

| Key | Where to get it |
|-----|----------------|
| Groq API key | [console.groq.com](https://console.groq.com) → API Keys |
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (enable Inference) |

## Run it locally

```bash
git clone https://github.com/auntor101/visual-mind-ai
cd visual-mind-ai

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```

Edit `.env` and add your keys:
```
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
```

Then run:
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) — choose **Docker** as the SDK
2. Push this repo to that Space
3. Go to **Settings → Secrets** and add `GROQ_API_KEY` and `HF_TOKEN`
4. Wait for the build — your app will be live at the Space URL

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [Groq](https://console.groq.com/) — Llama 4 Scout 17B vision model
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference) — BLIP-large captioning
- [Pillow](https://python-pillow.org/) — image processing

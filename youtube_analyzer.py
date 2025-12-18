import os
import re
import textwrap
from pytube import YouTube
import whisper
from transformers import pipeline

# =========================
# UTILIDADES
# =========================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("♪", "")
    return text.strip()

def split_text(text, max_size=1000):
    return textwrap.wrap(text, max_size)

def step(msg):
    print(f"\n🔹 {msg}...")

# =========================
# DOWNLOAD DO ÁUDIO
# =========================

def download_audio(url, output_dir="audio"):
    step("Baixando áudio do YouTube")
    ensure_dir(output_dir)

    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    file_path = audio.download(output_dir)

    base, _ = os.path.splitext(file_path)
    mp3_path = base + ".mp3"
    os.rename(file_path, mp3_path)

    return mp3_path

# =========================
# TRANSCRIÇÃO
# =========================

def transcribe_audio(audio_path):
    step("Transcrevendo áudio")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="pt")
    return clean_text(result["text"])

# =========================
# RESUMO
# =========================

def summarize_text(text):
    step("Gerando resumo detalhado")

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=180,
            min_length=80,
            do_sample=False
        )
        summaries.append(summary[0]["summary_text"])

    return " ".join(summaries)

# =========================
# ANALISADOR PRINCIPAL
# =========================

def analyze_youtube_video(url):
    audio = download_audio(url)
    transcription = transcribe_audio(audio)
    summary = summarize_text(transcription)

    return transcription, summary

# =========================
# EXECUÇÃO
# =========================

if __name__ == "__main__":
    print("\n📺 ANALISADOR DE VÍDEOS DO YOUTUBE")
    print("Cole o link do vídeo e receba um resumo completo.\n")

    link = input("🔗 Link do YouTube: ").strip()

    transcription, summary = analyze_youtube_video(link)

    print("\n" + "="*50)
    print("📌 RESUMO COMPLETO DO VÍDEO\n")
    print(summary)
    print("\n" + "="*50)

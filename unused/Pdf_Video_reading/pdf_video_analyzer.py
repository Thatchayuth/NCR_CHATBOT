import fitz  # สำหรับอ่าน PDF
import whisper
import soundfile as sf
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
import tempfile
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from moviepy.editor import VideoFileClip
import librosa
import torchaudio
import torch
import noisereduce as nr


# อ่านข้อความจาก PDF ภาษาไทย
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)




# def transcribe_video_url_with_transformers(url, model_name="scb10x/monsoon-whisper-medium-gigaspeech2"):
#     # 🔽 ดาวน์โหลดวิดีโอมาเป็นไฟล์ mp4
#     video_path = download_video_from_url(url)

#     # 🔄 แยกเสียงจาก .mp4 เป็น .wav
   
#     audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#     video = VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path, logger=None)  # mute log

#     # 🔄 โหลดเสียง
#     waveform, sample_rate = torchaudio.load(audio_path)
#     if sample_rate != 16000:
#         waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

#     # 🎯 เตรียม model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     processor = WhisperProcessor.from_pretrained(model_name)
#     model = WhisperForConditionalGeneration.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
#     )
#     model.to(device)
#     model.eval()

#     # ✨ ถอดเสียง
#     inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
#     with torch.no_grad():
#         predicted_ids = model.generate(inputs)
#         text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     return text

def transcribe_video_url_with_chunks(url, model_name="scb10x/monsoon-whisper-medium-gigaspeech2"):
    # 🔽 ดาวน์โหลดวิดีโอมาเป็นไฟล์ mp4
    video_path = download_video_from_url(url)

    # 🔄 แยกเสียงจาก .mp4 เป็น .wav
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)

    # 🔄 โหลดเสียง
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    # 🔄 แบ่งเสียงเป็น chunk ละ 30 วินาที (480000 sample ถ้า 16kHz)
    chunk_size = 30 * 16000
    texts = []

    for i in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, i:i+chunk_size]
        if chunk.shape[1] == 0:
            continue
        inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        inputs = inputs.to(device=device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(inputs)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            texts.append(text)

    return " ".join(texts)

# ถอดเสียงจากวิดีโอเป็นข้อความภาษาไทย
def download_video_from_url(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            tmp_file.write(chunk)
    tmp_file.close()
    return tmp_file.name

def transcribe_video_url(url):
    video_path = download_video_from_url(url)
    model = whisper.load_model("large",device='cuda')  # หรือ tiny, large ก็ได้
    result = model.transcribe(video_path, language="th")
    return result["text"]



def transcribe_video(video_path,language="th"):
    model = whisper.load_model("medium")  # หรือ tiny/large ก็ได้
    result = model.transcribe(video_path, language=language)
    return result["text"]


def transcribe_audio_file(audio_path, language="en"):
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def transcribe_long_audio_file(
    audio_path,
    language="en",
    model_size="medium",
    chunk_minutes=10,
    output_txt=None,
):
    """
    Transcribe a long audio file (e.g. 2 hours) by splitting into chunks.

    Args:
        audio_path   : path to the audio file (.m4a / .mp3 / .wav / etc.)
        language     : language code e.g. "en", "th"
        model_size   : whisper model size: tiny / base / small / medium / large
        chunk_minutes: how many minutes per chunk (default 10 min)
        output_txt   : optional path to save incremental transcription; if None,
                       saves next to the audio file with the same stem + ".txt"
    Returns:
        Full transcription as a single string.
    """
    import os
    import math
    import numpy as np
    from pydub import AudioSegment

    # ---- resolve output path ------------------------------------------------
    if output_txt is None:
        base = os.path.splitext(audio_path)[0]
        output_txt = base + "_transcript.txt"

    # ---- convert to wav via pydub (handles m4a / mp3 / ogg / etc.) ----------
    print(f"⏳ Loading audio: {audio_path}")
    ext = os.path.splitext(audio_path)[1].lstrip(".").lower() or "m4a"
    audio_seg = AudioSegment.from_file(audio_path, format=ext)
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

    # convert pydub samples → numpy float32 array
    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    samples /= float(2 ** (audio_seg.sample_width * 8 - 1))  # normalise to [-1, 1]
    sr = 16000
    y = samples

    total_seconds = len(y) / sr
    total_minutes = total_seconds / 60
    print(f"   Duration: {total_minutes:.1f} min  |  Sample rate: {sr} Hz")

    # ---- load whisper model --------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading Whisper model '{model_size}' on {device} ...")
    model = whisper.load_model(model_size, device=device)

    # ---- chunk parameters ----------------------------------------------------
    chunk_samples = chunk_minutes * 60 * sr
    n_chunks = math.ceil(len(y) / chunk_samples)
    print(f"📦 Splitting into {n_chunks} chunks of {chunk_minutes} min each\n")

    all_texts = []

    # Open output file in append mode so progress is saved after every chunk
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for idx in range(n_chunks):
            start = idx * chunk_samples
            end = min(start + chunk_samples, len(y))
            chunk = y[start:end]

            start_min = start / sr / 60
            end_min = end / sr / 60
            print(f"  [{idx+1}/{n_chunks}] Transcribing {start_min:.1f}–{end_min:.1f} min ...", end=" ", flush=True)

            # Write chunk to a temp wav file because whisper.transcribe expects a file path or numpy array
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, chunk, sr)

            try:
                result = model.transcribe(tmp_path, language=language, fp16=(device == "cuda"))
                text = result["text"].strip()
            finally:
                os.remove(tmp_path)

            all_texts.append(text)
            # Write timestamp header + text to the running output file
            f_out.write(f"[{start_min:.1f}–{end_min:.1f} min]\n{text}\n\n")
            f_out.flush()
            print(f"✅  ({len(text)} chars)")

    full_text = "\n".join(all_texts)
    print(f"\n✅ Done! Full transcript saved to: {output_txt}")
    return full_text



# สรุปข้อความโดย LLM ภาษาไทย
def summarize_text(text):
    llm = Ollama(model="llama3")  # ถ้ามี OpenThaiGPT ให้เปลี่ยนตรงนี้
    prompt = PromptTemplate.from_template(
        "สรุปเนื้อหาต่อไปนี้ให้เข้าใจง่าย โดยเรียบเรียงเป็นรายงาน:\n\n{content}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(content=text)

# เขียนสรุปลง Word ภาษาไทย
def write_to_word(text, filename):
    doc = Document()
    doc.add_heading("สรุปรายงาน", 0)
    doc.add_paragraph(text)
    doc.save(filename)

# 🧪 ทดลองใช้งาน (เลือกอย่างใดอย่างหนึ่ง)
# source_type = "video"  # หรือ "pdf"

# if source_type == "pdf":
#     content = extract_text_from_pdf("ตัวอย่าง.pdf")
# elif source_type == "video":
#     content = transcribe_video("วิดีโอตัวอย่าง.mp4")

# summary = summarize_text(content)
# write_to_word(summary, "รายงาน.docx")
# print("✅ เสร็จเรียบร้อย บันทึกลงไฟล์แล้ว: รายงาน.docx")

audio_file = r"C:\Users\Administrator\Desktop\Files\test.m4a"

# ---- For short files (< ~30 min) use the simple version --------------------
# english_text = transcribe_audio_file(audio_file, language="en")

# ---- For long files (e.g. 2 hours) use the chunked version -----------------
# Chunks are 10 min each; transcript is saved incrementally to a .txt file
# so you don't lose progress if something crashes.
english_text = transcribe_long_audio_file(
    audio_file,
    language="en",
    model_size="medium",   # change to "large" for better accuracy (needs more VRAM)
    chunk_minutes=10,       # adjust chunk size as needed
    # output_txt=r"C:\Users\Administrator\Desktop\Files\test_transcript.txt",
)
print("📄 Transcription:\n", english_text)

# #%%

# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"

# %%ุ

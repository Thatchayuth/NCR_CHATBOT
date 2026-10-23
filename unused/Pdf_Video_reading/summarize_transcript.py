import requests
import json
import re
import time
import sys
from datetime import datetime

# ========== CONFIG ==========
TRANSCRIPT_FILE = "test_transcript.txt"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "scb10x/typhoon2.5-qwen3-4b:latest"
CHUNK_SIZE = 3000        # ตัวอักษรต่อ chunk (ปรับได้)
OVERLAP = 200            # overlap ระหว่าง chunk เพื่อไม่ให้ context ขาด
OUTPUT_FILE = "summary_output.txt"
CHUNK_TIMEOUT = 120      # วินาที: timeout รวมต่อ chunk (ถ้าเกินนี้ถือว่าค้าง)
MAX_TOKENS = 800         # จำนวน token สูงสุดที่ให้ model generate ต่อ chunk
# ============================


def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """แบ่ง text เป็น chunk โดยตัดที่ newline ใกล้ end สุด แต่ไม่ใกล้ start เกินไป"""
    chunks = []
    start = 0
    min_chunk = chunk_size // 2  # chunk ต้องยาวอย่างน้อยครึ่งหนึ่งของ chunk_size
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # หา newline ใน zone ท้าย เพื่อไม่ให้ตัดใกล้หัว chunk
            cut = text.rfind("\n", start + min_chunk, end)
            if cut == -1:
                cut = end
        else:
            cut = text_len  # chunk สุดท้าย

        chunks.append(text[start:cut].strip())

        if cut >= text_len:
            break  # จบแล้ว ออกทันที ไม่ให้ overlap ดึงกลับมา

        next_start = cut - overlap
        if next_start <= start:  # ป้องกัน infinite loop
            next_start = cut
        start = next_start

    return [c for c in chunks if len(c) > 50]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def call_ollama_stream(prompt: str, model: str = MODEL,
                       timeout: int = CHUNK_TIMEOUT,
                       max_tokens: int = MAX_TOKENS) -> str:
    """เรียก Ollama แบบ stream พร้อม timeout และจำกัด token"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,   # จำกัด token ที่ generate
            "temperature": 0.3,          # ลด randomness ให้ตอบกระชับขึ้น
        }
    }
    result = []
    dot_count = 0
    token_count = 0
    deadline = time.time() + timeout
    try:
        # connect timeout=10s, read timeout per-token=30s
        with requests.post(OLLAMA_URL, json=payload, stream=True,
                           timeout=(10, 30)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                # ตรวจ wall-clock timeout
                if time.time() > deadline:
                    print(f"\n[WARN] Timeout after {timeout}s — truncating output",
                          flush=True)
                    break
                if line:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    result.append(token)
                    token_count += 1
                    dot_count += 1
                    if dot_count % 20 == 0:
                        elapsed = int(time.time() - (deadline - timeout))
                        print(f".({elapsed}s)", end="", flush=True)
                    if data.get("done", False):
                        break
                    # ถ้า generate เกิน max_tokens ให้หยุด
                    if token_count >= max_tokens:
                        print(f"\n[WARN] Reached max_tokens={max_tokens} — stopping",
                              flush=True)
                        break
        print()  # ขึ้นบรรทัดใหม่หลัง dots
        return "".join(result).strip()
    except requests.exceptions.ConnectionError:
        print()
        return "[ERROR] ไม่สามารถเชื่อมต่อ Ollama ได้ กรุณารัน: ollama serve"
    except requests.exceptions.ReadTimeout:
        print()
        partial = "".join(result).strip()
        log(f"[WARN] Read timeout — returning partial output ({len(partial)} chars)")
        return partial if partial else "[ERROR] Read timeout — no output"
    except Exception as e:
        print()
        return f"[ERROR] {e}"


def summarize_chunk(chunk: str, chunk_index: int, total: int) -> str:
    prompt = f"""You are a professional meeting summarizer. 
Summarize the following meeting transcript segment (part {chunk_index + 1} of {total}) in clear bullet points.
Focus on: key decisions, action items, issues raised, and important discussion points.
Be concise but capture all important details.

Transcript segment:
{chunk}

Summary:"""
    return call_ollama_stream(prompt)


def final_summary(partial_summaries: list[str]) -> str:
    BATCH_SIZE = 5          # รวมทีละ 5 parts เพื่อไม่ให้ prompt ใหญ่เกิน
    FINAL_TIMEOUT = 180     # timeout สำหรับ final summary ยาวกว่าปกติ
    FINAL_MAX_TOKENS = 1200

    # ถ้ามีหลาย parts → batch รวมทีละ BATCH_SIZE ก่อน
    if len(partial_summaries) > BATCH_SIZE:
        log(f"Too many parts ({len(partial_summaries)}) → batching into groups of {BATCH_SIZE} first...")
        batched = []
        for i in range(0, len(partial_summaries), BATCH_SIZE):
            batch = partial_summaries[i:i + BATCH_SIZE]
            combined = "\n\n".join([f"[Part {i+j+1}]\n{s}" for j, s in enumerate(batch)])
            prompt = f"""Summarize the following meeting notes into concise bullet points covering key decisions, action items, and issues:

{combined}

Concise summary:"""
            log(f"  Batching parts {i+1}–{min(i+BATCH_SIZE, len(partial_summaries))}...")
            print(f"         Batch response ", end="", flush=True)
            result = call_ollama_stream(prompt, timeout=FINAL_TIMEOUT, max_tokens=600)
            batched.append(result)
        partial_summaries = batched
        log(f"Batching done → {len(batched)} groups → generating final summary...")

    combined = "\n\n".join(
        [f"[Group {i+1}]\n{s}" for i, s in enumerate(partial_summaries)]
    )
    prompt = f"""You are a professional meeting summarizer.
Below are summarized notes from a business meeting.
Create a comprehensive final summary with these sections:
1. Main topics discussed
2. Key decisions made
3. Action items (and owners if mentioned)
4. Issues/concerns raised
5. Next steps

Meeting notes:
{combined}

Final comprehensive summary:"""
    print(f"         Final response ", end="", flush=True)
    return call_ollama_stream(prompt, timeout=FINAL_TIMEOUT, max_tokens=FINAL_MAX_TOKENS)


def check_ollama() -> bool:
    """ตรวจสอบว่า Ollama พร้อมใช้งานและ model มีอยู่"""
    try:
        log(f"Checking Ollama connection at {OLLAMA_URL}...")
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        model_found = any(MODEL.split(":")[0] in m for m in models)
        log(f"Ollama OK — available models: {models}")
        if not model_found:
            log(f"WARNING: model '{MODEL}' not found! Run: ollama pull {MODEL}")
            return False
        log(f"Model '{MODEL}' found — ready.")
        return True
    except Exception as e:
        log(f"ERROR: Cannot connect to Ollama — {e}")
        log("Please run 'ollama serve' in a separate terminal first.")
        return False


def main():
    total_start = time.time()
    log(f"Loading transcript: {TRANSCRIPT_FILE}")
    text = load_transcript(TRANSCRIPT_FILE)
    log(f"Total characters: {len(text):,}")

    chunks = split_into_chunks(text, CHUNK_SIZE, OVERLAP)
    log(f"Split into {len(chunks)} chunks (size ~{CHUNK_SIZE} chars each)")
    log(f"Model: {MODEL}")
    log(f"Output file: {OUTPUT_FILE}")
    print("-" * 60)

    if not check_ollama():
        sys.exit(1)

    partial_summaries = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=" * 60 + "\n")
        out.write("TRANSCRIPT SUMMARY REPORT\n")
        out.write("=" * 60 + "\n\n")

        # --- Summarize each chunk ---
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            log(f"[{i+1}/{len(chunks)}] Summarizing chunk... ({len(chunk):,} chars)")
            print(f"         Waiting for model response ", end="", flush=True)

            summary = summarize_chunk(chunk, i, len(chunks))
            elapsed = time.time() - chunk_start

            partial_summaries.append(summary)
            out.write(f"--- Part {i+1}/{len(chunks)} ---\n")
            out.write(summary)
            out.write("\n\n")
            out.flush()

            log(f"[{i+1}/{len(chunks)}] Done in {elapsed:.1f}s — saved to file")
            print("-" * 40)

        # --- Final consolidated summary ---
        log(f"Generating final consolidated summary from {len(chunks)} parts...")
        print(f"         Waiting for model response ", end="", flush=True)
        final_start = time.time()
        final = final_summary(partial_summaries)
        log(f"Final summary done in {time.time() - final_start:.1f}s")

        out.write("=" * 60 + "\n")
        out.write("FINAL CONSOLIDATED SUMMARY\n")
        out.write("=" * 60 + "\n\n")
        out.write(final)
        out.write("\n")

    total_elapsed = time.time() - total_start
    print("=" * 60)
    log(f"All done! Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    log(f"Output saved to: {OUTPUT_FILE}")
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(final[:1000] + "..." if len(final) > 1000 else final)


if __name__ == "__main__":
    main()

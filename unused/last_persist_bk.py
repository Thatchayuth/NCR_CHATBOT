import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.embeddings import OllamaEmbeddings
# โหลด JSON
with open("data_with_keywords_output_1_updated.json", "r", encoding="utf-8") as f: 
    data = json.load(f)

# เตรียม Document

def to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed]
            except Exception:
                pass
        # Split by comma and strip whitespace, filter out empty
        return [v.strip() for v in val.split(",") if v.strip()]
    return []

documents = []
for item in data:
#     content = f"""หัวข้อ: {item.get('topic', '')}
# อุปกรณ์: {item.get('hardware', '')}
# ประเภท: {item.get('type', '')}
# สรุป: {item.get('summarize', '')}
# คำค้นภาษาไทย: {', '.join(to_list(item.get('keyword_th', [])))}
# คำค้นภาษาอังกฤษ: {', '.join(to_list(item.get('keyword_en', [])))}
# คำสำคัญอื่นๆ: {item.get('extra_keywords', '')}
# สรุปย่อ: {item.get('pdf_summary', '')}
# """
    content = f"""หัวข้อ: {item.get('topic', '')}
สรุปย่อ: {item.get('pdf_summary', '')}
"""

    doc = Document(
        page_content=content,
        metadata={
            "topic": item.get("topic", ""),
            "hardware": item.get("hardware", ""),
            "type": item.get("type", ""),
            "urlfilenew": item.get("urlfilenew", ""),
            "keyword_th": ", ".join(to_list(item.get("keyword_th", []))),
            "keyword_en": ", ".join(to_list(item.get("keyword_en", []))),
            "extra_keywords": item.get("extra_keywords", ""),
            "clip": item.get("clip", ""),
            "url": item.get("urlfilenew", ""),
            'pdf_summary': item.get('pdf_summary', ''),
        }
    )
    documents.append(doc)

# บันทึกเข้า Chroma
embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device": "cuda"})
# embedding = OllamaEmbeddings(
#     # model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
#     model="nomic-embed-text",
#     base_url="http://127.0.0.1:11434"6
# )
# embedding = OllamaEmbeddings(
#     # model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
#     model="dengcao/Qwen3-Embedding-8B:Q4_K_M",
#     base_url="http://127.0.0.1:11434"
#)


db = Chroma.from_documents(documents, embedding=embedding, persist_directory="last_persist_db_qwen_ollama_rev2")
db.persist()


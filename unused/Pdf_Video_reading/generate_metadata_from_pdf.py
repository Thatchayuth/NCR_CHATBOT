#%%
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
import requests
from io import BytesIO
import tempfile
import json
from langchain_community.llms import Ollama



def get_text_from_url_pdf(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่าดาวน์โหลดสำเร็จ

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        return "\n".join(doc.page_content for doc in documents)

    except Exception as e:
        print(f"❌ Error reading PDF from URL: {e}")
        return ""
    
llm = Ollama(model="scb10x/llama3.1-typhoon2-8b-instruct:latest")

def generate_keywords_from_text(text):
    prompt = f"""
    สร้างคีย์เวิด 5-10 คำเป็นภาษาไทยและภาษาอังกฤษ และคำทับศัพท์(ถ้ามี) และสรุปข้อความ(summmarize)ต่อไปนี้:
    🔒 ตอบกลับเป็น JSON เท่านั้น ห้ามมีคำอธิบายเพิ่มเติมใด ๆ
    🔁 หากไม่สามารถสรุปหรือแยก keyword ได้ ให้ใช้ค่าว่าง
    {{
     'summarize':"..............................." ,
    'keyword_th':".....,......,....,...."  ,
    'keyword_en':"...,....,...,..." 
    }}
    กติกา
    *** ถ้าคิดว่าข้อความเกี่ยวกับการดูแลรักษาอุปกรณ์ , ให้ใส่  keyword 'P2A' และ 'พีทูเอ'.
    *** ตอบเฉพาะ คีย์เวิดในรูปแบบ json เท่านั้น
    *** ห้ามอธิบาย .
    *** ห้ามมีประโยคอื่่นๆนอกจาก json
    

    ข้อความ: {text}

    คำตอบในรูปแบบ json:
    """
    return llm.invoke(prompt).strip()

def fix_and_extract_generate_field(data):
    try:
        gen = data.get("generate", "")
        # แก้รูปแบบ JSON ที่ผิด (เช่น ใช้ single quote)
        gen_fixed = gen.replace("'", "\"")
        parsed = json.loads(gen_fixed)

        # แปลง keyword_th / keyword_en จาก string → list
        def parse_keywords(kw_str):
            return [k.strip().strip("\"'") for k in kw_str.split(",") if k.strip()]

        data["summary"] = parsed.get("summarize", "")
        data["keyword_th"] = parse_keywords(parsed.get("keyword_th", ""))
        data["keyword_en"] = parse_keywords(parsed.get("keyword_en", ""))

    except Exception as e:
        print(f"❌ Error fixing generate field: {e}")
        data["summary"] = ""
        data["keyword_th"] = []
        data["keyword_en"] = []

    return data


# 1. Load your file
with open("data_with_new_keywords_1.json", "r", encoding="utf-8") as f:
    datas = json.load(f)
# %%
new_data =[]
for data in datas:
    print(data)
    url = data.get('urlfilenew','')
    if url:
        pdf_text = get_text_from_url_pdf(url)
        print(pdf_text)
        if pdf_text:    
            ans_from_llm = generate_keywords_from_text(pdf_text)
            print(ans_from_llm)
            if ans_from_llm:
                data['generate']= ans_from_llm
                # data = fix_and_extract_generate_field(data)
        new_data.append(data)        
    else:
        new_data.append(data) 
        continue
#%%


# for item in data_list:  # data_list คือ list ที่คุณโหลดจาก JSON file
#     item = fix_and_extract_generate_field(item)

# %%
with open("data_with_keywords_output_1.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

# %%

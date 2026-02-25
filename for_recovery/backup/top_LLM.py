from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

_embedding = None
_vectordb = None
_llm = None
persist_directory = "last_persist_db_qwen_ollama_rev3"
all_link = []
def get_llm():
    global _llm
    if _llm is None:
        print("🔄 โหลด LLM ครั้งแรก...")
        _llm = Ollama(
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
            # model = "qwen3:0.6b",
            # model = "llama3:8b",
            # model = "llama3.1:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0.1,
            top_p=0.7,
            top_k=60, 
            num_predict= 300,
            num_ctx = 5000,
            # base_url="https://89kpfn5l-11434.asse.devtunnels.ms"
        )
    return _llm

def get_embedding():
    global _embedding
    if _embedding is None:
        print("🔄 โหลด embedding model ครั้งแรก...")
        _embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device": "cuda"})
        print('✅ embedding model พร้อมใช้งาน')
    return _embedding

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        print("🔄 โหลด vector database ครั้งแรก...")
        _vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=get_embedding()
        )
    return _vectordb

def vector_search(query, k=10):
    """
    Perform a vector search in the database.
    :param query: The query string to search for.
    :param k: The number of results to return.
    :return: List of documents matching the query.
    """
    vectordb = get_vectordb()
    results = vectordb.similarity_search(query, k=k)
    return results

def match_query_with_vector_db(query, results):
    """
    Match a query with the vector database.
    :param query: The query string to match.
    :param results: List of search results to filter.
    :return: List of documents matching the query.
    """
    llm = get_llm()
    
    # ตรวจสอบความต้องการไฟล์
    check_need_template = f"""วิเคราะห์คำถามนี้ว่าผู้ใช้ต้องการไฟล์ประเภทใดหรือไม่:

คำถาม: "{query}"

พิจารณาจากคำเหล่านี้:
- ต้องการ PDF: "ขอไฟล์", "ขอ PDF", "ส่งไฟล์", "ดาวน์โหลด", "เอกสาร"
- ต้องการ VDO: "ขอวีดีโอ", "ขอคลิป", "ดูวิดีโอ", "วีดีโอสอน", "คลิปสอน"
- ไม่ต้องการไฟล์: ถามหาข้อมูล คำอธิบาย วิธีทำ โดยไม่ได้เอ่ยถึงไฟล์

ตอบเพียงคำเดียว:
- PDF (ถ้าต้องการไฟล์ PDF)
- VDO (ถ้าต้องการวีดีโอ)  
- ไม่ต้องการ (ถ้าแค่ต้องการคำอธิบาย/ข้อมูล)

คำตอบ:"""
    user_need = llm(check_need_template)
    user_need = user_need.strip().lower()
    print(user_need)
    if user_need == "pdf":
        print("🔄 ค้นหาไฟล์ PDF ที่เกี่ยวข้องกับคำถาม...")
        results = [result for result in results if result.metadata.get("have_pdf", "no") == "yes"]
    elif user_need == "vdo":
        print("🔄 ค้นหาวีดีโอคลิปที่เกี่ยวข้องกับคำถาม...")
        results = [result for result in results if result.metadata.get("have_clip", "no") == "yes"]
    else:
        print("🔄 ไม่ต้องการไฟล์ PDF หรือ วีดีโอคลิป")

    # ✅ แก้ไข: ใช้ template parameter
    prompt_template = PromptTemplate(
        input_variables=["question", "topic", "content_preview"],
        template="""ตรวจสอบความเกี่ยวข้องระหว่างคำถามและเนื้อหา:
    
คำถาม: {question}

หัวข้อ: {topic}
เนื้อหา: {content_preview}

ตอบแค่ YES หรือ NO:
- YES = เนื้อหานี้ตอบคำถามได้หรือเกี่ยวข้องกับคำถาม
- NO = เนื้อหานี้ไม่เกี่ยวข้องกับคำถาม

คำตอบ:"""
    )
    
    related_docs = []
    
    print(f"🔍 กำลังตรวจสอบความเกี่ยวข้องของ {len(results)} รายการ...")
    
    if len(results) <=2 :
        return results
    else:
        for i, result in enumerate(results, 1):
            print(f"   📄 ตรวจสอบรายการที่ {i}/{len(results)}: {result.metadata.get('topic', 'ไม่มีหัวข้อ')+result.metadata.get('pdf_summary', result.metadata.get('summarize', '')) }...")

            # ใช้ LLM ตรวจสอบความเกี่ยวข้อง
            try:
                # สร้าง prompt จาก template
                prompt = prompt_template.format(
                    question=query,
                    topic=result.metadata.get("topic", "ไม่มีหัวข้อ"),
                    content_preview=result.metadata.get("pdf_summary",result.metadata.get('summarize', ""))
                )
                
                response = llm(prompt)
                response = response.strip().lower()
                
                if "yes" in response:
                    related_docs.append(result)
                    print(f"      ✅ เกี่ยวข้อง - เก็บไว้")
                else:
                    print(f"      ❌ ไม่เกี่ยวข้อง - ตัดออก")
                    
            except Exception as e:
                print(f"      ⚠️ Error checking relevance: {e}")
                # หากเกิด error ให้เก็บไว้ก่อน
                related_docs.append(result)
                
        return related_docs

def generate_final_answer(query, related_docs):
    """สร้างคำตอบสุดท้ายจากเอกสารที่เกี่ยวข้อง"""
    llm = get_llm()
    
    if not related_docs:
        ans = f"ขอโทษครับ ไม่พบข้อมูลที่เกี่ยวข้องกับ{query}ของคุณ มีความเป็นไปได้ว่า \nคำถาม:{query}\n อาจะไม่มีในคลังความรู้ของผม  กรุณาติดต่อฝ่าย ICT ได้ที่เบอร์ 601(ระยอง) หรือ 116(กรุงเทพ) เพื่อขอความช่วยเหลือเพิ่มเติม"
        llm_creative = Ollama(
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
            base_url="http://127.0.0.1:11434",
            temperature=0.9,  # เพิ่ม temperature เพื่อความหลากหลาย
            top_p=0.9,
            top_k=60,
            num_predict=300,
            num_ctx=5000,
        )
        answer_llm = llm_creative("สร้างประโยคนี้ใหม่ให้ดูดี ประโยค:"+ans+"ประโยคใหม่ :")
        return answer_llm
    
    # รวมเนื้อหาจากเอกสารที่เกี่ยวข้อง
    content_parts = []
    resources = []
    
    for i, doc in enumerate(related_docs, 1):
        content_parts.append(f"ข้อมูลที่ {i}: {doc.page_content}")
        
        # รวบรวม resources
        if doc.metadata.get("urlfilenew"):
            resources.append(f"📄 PDF: {doc.metadata.get('urlfilenew')}")
        if doc.metadata.get("clip"):
            resources.append(f"🎥 Video: {doc.metadata.get('clip')}")
    
    combined_content = "\n\n".join(content_parts)
    
    # สร้าง prompt สำหรับการตอบคำถาม
    answer_prompt = f"""คุณคือ i-Bot เด็กชายอายุ 18 ปี แผนก ICT บริษัท NCR Rubber
คุณเป็นผู้ช่วยด้าน IT ที่เป็นมิตรและให้ความช่วยเหลือ

กฎการตอบ:
1. ใช้ "ครับ" และ "นะครับ" ในทุกประโยค
2. ตอบให้กระชับ เข้าใจง่าย
3. ใช้ข้อมูลจากเนื้อหาที่ให้มาในการตอบ
4. หากมีขั้นตอนการแก้ไข ให้เรียงลำดับชัดเจน

คำถาม: {query}

เนื้อหาที่เกี่ยวข้อง:
{combined_content}

ถ้าไม่เจอคำตอบในเนื้อหา ให้ตอบว่า "ขอโทษครับ ไม่พบข้อมูลที่เกี่ยวข้องกับ คำถาม:{query}ของคุณ คุณสามารถติดต่อฝ่าย ICT ได้ที่เบอร์ 601(ระยอง) หรือ 116(กรุงเทพ) เพื่อขอความช่วยเหลือเพิ่มเติม"
คำตอบ:"""
    
    try:
        answer = llm(answer_prompt)
        
        # เพิ่ม resources ถ้ามี
        if resources:
            answer += "\n\n📎 ลิงก์ที่เกี่ยวข้อง:\n" + "\n".join(resources)
            
        return answer.strip()
        
    except Exception as e:
        return f"ขอโทษครับ เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"

def remove_duplicate_resources(docs):
    """ตัดข้อมูลที่มี urlfilenew หรือ clip ซ้ำออก"""
    seen_urls = set()
    seen_clips = set()
    unique_docs = []
    
    for doc in docs:
        url = doc.metadata.get('urlfilenew', '')
        clip = doc.metadata.get('clip', '')
        
        # ตรวจสอบ URL ซ้ำ
        is_duplicate = False
        if url and url in seen_urls:
            print(f"   ⚠️ ตัดข้อมูลซ้ำ (URL): {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
            is_duplicate = True
        
        # ตรวจสอบ Clip ซ้ำ
        if clip and clip in seen_clips:
            print(f"   ⚠️ ตัดข้อมูลซ้ำ (Video): {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
            is_duplicate = True
        
        if not is_duplicate:
            unique_docs.append(doc)
            if url:
                seen_urls.add(url)
            if clip:
                seen_clips.add(clip)
    
    return unique_docs

def collect_unique_links(related_docs):
    """เก็บลิงก์ PDF และ Video แค่อันละ 1 ตัว"""
    global all_link
    all_link = []
    pdf_link = None
    video_link = None
    
    # หาลิงก์แรกของแต่ละประเภท
    for doc in related_docs:
        # เก็บ PDF link แรกที่เจอ
        if not pdf_link and doc.metadata.get('urlfilenew'):
            pdf_link = doc.metadata.get('urlfilenew')
            print(f"   📄 เลือก PDF: {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
        
        # เก็บ Video link แรกที่เจอ
        if not video_link and doc.metadata.get('clip'):
            video_link = doc.metadata.get('clip')
            print(f"   🎥 เลือก Video: {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
        
        # ถ้าได้ครบทั้งสองแล้ว ก็หยุด
        if pdf_link and video_link:
            break
    
    # สร้าง link object
    if pdf_link or video_link:
        link = {
            'link_pdf': pdf_link if pdf_link else "",
            'link_clip': video_link if video_link else ""
        }
        all_link.append(link)
        print(f"   🔗 เพิ่มลิงก์: PDF={'มี' if pdf_link else 'ไม่มี'}, Video={'มี' if video_link else 'ไม่มี'}")
    
    return pdf_link, video_link

def process_query(query):
    global all_link
    """ฟังก์ชันประมวลผลคำถาม"""
    if not query.strip():
        print("⚠️ กรุณาใส่คำถามครับ")
        return
    
    try:
        print("🔄 เริ่มค้นหาในฐานข้อมูล...")
        results = vector_search(query, k=10)
        print(f"✅ พบเอกสาร {len(results)} รายการจากการค้นหา")
        
        print("\n🤖 กำลังใช้ LLM กรองข้อมูล...")
        related_docs = match_query_with_vector_db(query, results)
        print(f"🎯 พบเอกสารที่เกี่ยวข้อง {len(related_docs)} รายการ")
        
        related_docs = remove_duplicate_resources(related_docs)
        if related_docs:
            print("\n📋 รายการเอกสารที่เกี่ยวข้อง:")
            for i, doc in enumerate(related_docs, 1):
                print(f"   {i}. {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
        pdf_link, video_link = collect_unique_links(related_docs)
      
        print("\n🤖 กำลังสร้างคำตอบ...")
        final_answer = generate_final_answer(query, related_docs)
        
        print("\n" + "="*60)
        print("🤖 i-Bot:")
        print(final_answer)
        print("="*60)
        data = {
            'answer':'',
            'link': all_link
        }
        return final_answer , data
        
    except Exception as e:
        error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
        print(error_msg)
        return error_msg , ''


if __name__ == "__main__":
    text = ""
    while text != 'y':
        text = input("🔄 กด Enter เพื่อเริ่มการค้นหา หรือพิมพ์ 'y' เพื่อออก: ")
        print(process_query(text))
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

_embedding = None
_vectordb = None
_vectordb_phone = None
_llm = None
persist_directory = "last_persist_db_qwen_ollama_rev3"
persist_directory_phone = "phone_directory_vectordb"
all_link = []
llm_model ="scb10x/llama3.1-typhoon2-8b-instruct:latest"
base_url="http://127.0.0.1:11434"

define_intension_user_prompt = """
แบ่งประเภทคำถามของผู้ใช้เป็น 3 ประเภทหลัก:
1. คำทักทายทั่วไป,คำขอบคุณ,คำถามที่ไม่ต้องการการแก้ปัญหา,คำถามที่ถามว่าทำอะไรได้บ้าง (ไม่ต้องการความช่วยเหลือ)
2. ถามหาเบอร์โทรศัพท์หรือข้อมูลการติดต่อของพนักงานในบริษัท NCR-Rubber (ต้องการข้อมูลการติดต่อ)
3. คำถามเกี่ยวกับการใช้งานคอมพิวเตอร์ หรือ ข้อมูล P2A , i-FAQ,AX4,Tip dd by TC,ICT rule (ต้องการข้อมูลหรือวิธีแก้ปัญหา)
ตอบแค่ 1 หรือ 2 หรือ 3 เท่านั้น 
คำถาม: """

talking_prompt = """ system_promt: 
                    -คุณคือi-Botเด็กชายอายุ 18 ปี แผนกICTของบริษัท NCR-Rubber ต้องไม่พูดถึง Typhoon เลย
                    -คุณมีความสามารถในการตอบคำถามเกี่ยวกับการใช้งานคอมพิวเตอร์และเทคโนโลยี
                    -ใช้คำว่า "ครับ" และ "นะครับ" ในทุกประโยค
                    -จะไม่พูดคำว่าค่ะ
                    ทำอะไรได้บ้าง:
                        - สามารถตอบคำถามที่พบบ่อยๆได้ (P2A,i-FAQ)
                        - สามารถสอบถามเบอร์โทรภายในองค์กรได้
                        - สามารถพูดคุยเล่นได้นิดหน่อย
Goal:
    -พยายามชวนให้คุยกับผู้ใช้ให้มากที่สุดเท่าที่จะทำได้
    -พยายามให้ผู้ใช้รู้สึกสบายใจและเป็นกันเองห
    -พยายามถามเรื่องเกี่ยวกับงานที่ผู้ใช้ทำอยู่(อาจจะไม่เกี่ยวกับ ICT ก็ได้)
    -ตอบกลับเป็นภาษาไทย นอกจากว่าผู้ใช้จะถามเป็นภาษาอังกฤษ
    -
query : """

define_user_need_in_rag = """
    แบ่งประเภทคำถามของผู้ใช้เป็น 2 อย่าง:
    1. ผู้ใชต้องการ สารบัญ, หัวข้อทั้งหมด
    2. ผู้ใช้ ไม่ต้องการ สารบัญ, หัวข้อทั้งหมด
    ******ตอบแค่ 1 หรือ 2 เท่านั้น
    คำถาม: """
def get_llm():
    global _llm
    if _llm is None:
        print("🔄 โหลด LLM ครั้งแรก...")
        _llm = Ollama(
            model= llm_model,
            # model = "qwen3:0.6b",
            # model = "llama3:8b",
            # model = "llama3.1:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0.1,
            top_p=0.3,
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

def get_vectordb_phone():
    global _vectordb_phone
    if _vectordb_phone is None:
        print("🔄 โหลด vector database ครั้งแรก...")
        _vectordb_phone = Chroma(
            persist_directory=persist_directory_phone,
            embedding_function=get_embedding()
        )
    return _vectordb_phone

def vector_search(query, k=10,mode =2,search=None):
    """
    Perform a vector search in the database.
    :param query: The query string to search for.
    :param k: The number of results to return.
    :return: List of documents matching the query.
    """
    if mode == 2:
        vectordb = get_vectordb()
        results = vectordb.similarity_search(query, k=k)
        return results
    else:
        vectordb = get_vectordb()
        results = vectordb.similarity_search('', k=k,filter={
        "folder": search}
)
        return results
def vector_search_phone(query, k=3):
    """
    Perform a vector search in the phone directory database.
    :param query: The query string to search for.
    :param k: The number of results to return.
    :return: List of documents matching the query.
    """
    if 'ทั้งหมด' in query:
        k = 10
    vectordb_phone = get_vectordb_phone()
    results = vectordb_phone.similarity_search(query, k=k)
    return results

def enhanced_phone_search(query, k=3):
    """
    Enhanced phone number search with better query processing for NCR phone directory
    """
    print(f"🔍 ค้นหาเบอร์โทรศัพท์: {query}")
    
    # Clean and prepare search terms
    phone_keywords = ['เบอร์', 'โทร', 'ติดต่อ', 'เบอร์โทร', 'เบอร์ติดต่อ', 'หมายเลข', 'โทรศัพท์','ขอ','พี่','น้อง','คุณ','ฝ่าย','เบอร์ของ','เบอร์ของคุณ']
    if 'ทั้งหมด' in query or 'แผนก' in query or 'ฝ่าย' in query:
        k= 10
    # Remove phone keywords to get the actual search term
    clean_query = query
    for keyword in phone_keywords:
        clean_query = clean_query.replace(keyword, ' ')
    
    clean_query = clean_query.strip()
    
    # Try different search approaches
    vectordb_phone = get_vectordb_phone()
    results = []
    
    # Method 1: Direct search
    try:
        direct_results = vectordb_phone.similarity_search(query, k=k)
        results.extend(direct_results)
    except Exception as e:
        print(f"⚠️ Direct search error: {e}")
    
    # Method 2: Clean query search
    if clean_query:
        try:
            clean_results = vectordb_phone.similarity_search(clean_query, k=k)
            results.extend(clean_results)
        except Exception as e:
            print(f"⚠️ Clean query search error: {e}")
    
    # Method 3: Search by individual words
    words = clean_query.split()
    for word in words:
        if len(word) > 2:  # Skip very short words
            try:
                word_results = vectordb_phone.similarity_search(word, k=k//2)
                results.extend(word_results)
            except Exception as e:
                print(f"⚠️ Word search error for '{word}': {e}")
    
    # Remove duplicates while preserving order
    unique_results = []
    seen = set()
    for result in results:
        result_key = result.page_content
        if result_key not in seen:
            unique_results.append(result)
            seen.add(result_key)
    
    return unique_results[:k]

def format_phone_results(results):
    """
    Format phone search results into a readable format for NCR directory
    """
    if not results:
        return "ไม่พบข้อมูลการติดต่อที่ค้นหา"
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        # Extract information from metadata
        metadata = result.metadata
        
        # Build contact info based on your table structure
        contact_info = []
        
        # Name (ชื่อ + นามสกุล)
        first_name = metadata.get('ชื่อ', '')
        last_name = metadata.get('นามสกุล', '')
        nickname = metadata.get('ชื่อเล่น', '')
        
        if first_name or last_name:
            name_parts = [first_name, last_name]
            full_name = ' '.join([n for n in name_parts if n])
            if nickname:
                full_name += f" ({nickname})"
            contact_info.append(f"👤 ชื่อ: {full_name}")
        
        # Position (ตำแหน่ง)
        position = metadata.get('ตำแหน่ง', '')
        if position:
            contact_info.append(f"💼 ตำแหน่ง: {position}")
        
        # Department (ฝ่าย)
        department = metadata.get('ฝ่าย', '') or metadata.get('Department', '')
        if department:
            contact_info.append(f"🏢 ฝ่าย: {department}")
        
        # Section
        section = metadata.get('Section', '')
        if section:
            contact_info.append(f"📂 แผนก: {section}")
        
        # Site
        site = metadata.get('Site', '')
        if site:
            contact_info.append(f"🏭 สถานที่: {site}")
        
        # Phone number (เบอร์โทร)
        phone = metadata.get('เบอร์โทร', '')
        if phone:
            contact_info.append(f"📞 เบอร์โทร: {phone}")
        
        # Emergency number (เบอร์ด่วน)
        emergency = metadata.get('เบอร์ด่วน', '')
        if emergency:
            contact_info.append(f"🚨 เบอร์ด่วน: {emergency}")
        
        # Shared usage (ใช้ร่วมกัน)
        shared = metadata.get('ใช้ร่วมกัน', '')
        if shared:
            contact_info.append(f"👥 ใช้ร่วมกัน: {shared}")
        
        if contact_info:
            formatted_results.append(f"📋 ผลลัพธ์ที่ {i}:\n" + "\n".join(contact_info))
    
    return "\n\n".join(formatted_results)

def generate_phone_answer(query, results):
    """
    Generate a natural answer for phone search results
    """
    if not results:
        return """ขอโทษครับ ผมไม่พบข้อมูลการติดต่อที่คุณต้องการค้นหา

🔍 ลองใช้คำค้นหาที่ชัดเจนขึ้น เช่น:
• ชื่อคน: "เบอร์ของคุณทองสร้อย" หรือ "เบอร์พี่่ก้อย"
• ชื่อฝ่าย: "เบอร์ฝ่ายธุรการ"
• ตำแหน่ง: "เบอร์ผู้จัดการ"
• สถานที่: "เบอร์ระยอง" หรือ "เบอร์กรุงเทพ"

หรือติดต่อสอบถามได้ที่:
📞 เบอร์ 601 (ระยอง) หรือ 116 (กรุงเทพ) นะครับ"""
    
    # Format the results
    formatted_results = format_phone_results(results)
    
    # Generate natural response
    llm = get_llm()
    
    phone_answer_prompt = f"""คุณคือ i-Bot ผู้ช่วยด้าน IT ที่เป็นมิตร
กรุณาสร้างคำตอบเกี่ยวกับข้อมูลการติดต่อให้เป็นธรรมชาติ

กฎการตอบ:
1. ใช้ "ครับ" และ "นะครับ" อย่างเหมาะสม
2. แสดงข้อมูลให้ชัดเจนและอ่านง่าย
3. เป็นมิตรและให้ความช่วยเหลือ
4. ถ้ามีหลายผลลัพธ์ให้อธิบายด้วย
5. แสดงชือจริง ชื่อเล่น เบอร์โทร เบอร์ติดต่อด้วยเสมอ (สำคัญมาก)
6. ถ้าในข้อมูลที่พบไม่มีข้อมูลที่ใกล้เคียงเลย ให้ตอบว่า "ลองค้นหาใหม่อีกครั้ง อาจจะลองบอกชื่อแผนกที่ คุณหาอยู่ก็ได้"

คำถาม: {query}

ข้อมูลที่พบ:
{formatted_results}

สร้างคำตอบที่เป็นธรรมชาติและเป็นมิตรครับ:"""
    
    try:
        answer = llm(phone_answer_prompt)
        return answer.strip()
    except Exception as e:
        # Fallback to formatted results if LLM fails
        return f"ผมพบข้อมูลการติดต่อที่คุณต้องการแล้วครับ:\n\n{formatted_results}"

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
    clean_query = llm(f"""ทำความสะอาดคำถามนี้ให้กระชับและเข้าใจง่ายโดยพยายามตัดพวก PDF หรีอ VDO ออก คำถาม: "{query} """)
    print(clean_query)
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
    
    for i, result in enumerate(results, 1):
        print(f"   📄 ตรวจสอบรายการที่ {i}/{len(results)}: {result.metadata.get('topic', 'ไม่มีหัวข้อ')+result.metadata.get('pdf_summary', result.metadata.get('summarize', '')) }...")

        # ใช้ LLM ตรวจสอบความเกี่ยวข้อง
        try:
            # สร้าง prompt จาก template
            prompt = prompt_template.format(
                question=clean_query,
                topic=result.metadata.get("topic", "ไม่มีหัวข้อ"),
                content_preview=result.metadata.get("pdf_summary", result.metadata.get("summary", result.page_content[:300]))
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

def extract_non_related_info(non_related_docs):
    """แยกข้อมูลจาก non_related_docs ให้เป็นระเบียบ"""
    if not non_related_docs:
        return {"topics": "", "links": "", "clips": ""}
    
    # รวบรวมข้อมูลและตัดซ้ำ
    topics = []
    links = []
    clips = []
    
    seen_topics = set()
    seen_links = set()
    seen_clips = set()
    
    for doc in non_related_docs:
        # เก็บ topic (ไม่ซ้ำ)
        topic = doc.metadata.get("topic", "ไม่มีหัวข้อ")
        if topic not in seen_topics:
            topics.append(f"• {topic}")
            seen_topics.add(topic)
        
        # เก็บ link (ไม่ซ้ำ)
        link = doc.metadata.get("urlfilenew", "")
        if link and link not in seen_links:
            links.append(f"📄 {link}")
            seen_links.add(link)
        
        # เก็บ clip (ไม่ซ้ำ)
        clip = doc.metadata.get("clip", "")
        if clip and clip not in seen_clips:
            clips.append(f"🎥 {clip}")
            seen_clips.add(clip)
    
    return {
        "topics": "\n".join(topics) if topics else "",
        "links": "\n".join(links) if links else "",
        "clips": "\n".join(clips) if clips else ""
    }

def generate_fallback_answer(query, non_related_info, llm):
    """สร้างคำตอบสำรองเมื่อไม่พบข้อมูลที่เกี่ยวข้อง"""
    
    # ✅ สร้างข้อความสำรองที่ดีขึ้น
    fallback_parts = [
        f"ขอโทษครับ ผมไม่พบข้อมูลที่เกี่ยวข้องโดยตรงกับ '{query}' ของคุณ"
    ]
    
    # เพิ่มหัวข้อที่อาจเกี่ยวข้อง
    if non_related_info["topics"]:
        fallback_parts.append(f"\nแต่ผมพบหัวข้อที่อาจเกี่ยวข้อง:\n{non_related_info['topics']}")
    
    # เพิ่มลิงก์และคลิป
    resources = []
    if non_related_info["links"]:
        resources.append(f"ลิงก์เอกสาร:\n{non_related_info['links']}")
    if non_related_info["clips"]:
        resources.append(f"วีดีโอ:\n{non_related_info['clips']}")
    
    if resources:
        fallback_parts.append(f"\nแหล่งข้อมูลที่อาจช่วยได้:\n" + "\n\n".join(resources))
    
    # เพิ่มข้อมูลติดต่อ
    fallback_parts.append(
        "\nหากต้องการความช่วยเหลือเพิ่มเติม กรุณาติดต่อฝ่าย ICT:\n"
        "📞 เบอร์ 601 (ระยอง) หรือ 116 (กรุงเทพ)"
    )
    
    raw_answer = "\n".join(fallback_parts)
    
    # ✅ ปรับปรุงการใช้ LLM ให้เป็นธรรมชาติมากขึ้น
    improvement_prompt = f"""
คุณคือ i-Bot ผู้ช่วยด้าน IT ที่เป็นมิตร กรุณาปรับปรุงข้อความนี้ให้:
1. อ่านง่ายและเป็นธรรมชาติ
2. ใช้คำว่า "ครับ" และ "นะครับ" อย่างเหมาะสม  
3. เป็นมิตรและให้กำลังใจ
4. รักษาข้อมูลสำคัญไว้ครบถ้วน

ข้อความต้นฉบับ:
{raw_answer}

ข้อความที่ปรับปรุงแล้ว:"""
    
    try:
        llm_creative = Ollama(
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
            base_url="http://127.0.0.1:11434",
            temperature=0.3,  # เพิ่มความเป็นธรรมชาติ
            top_p=0.8,
            top_k=200,
            num_predict=600,
            num_ctx=8000,
        )
        improved_answer = llm_creative(improvement_prompt)
        return improved_answer.strip()
    except Exception as e:
        print(f"⚠️ Error improving fallback answer: {e}")
        return raw_answer

def generate_final_answer(query, related_docs,non_related_docs):
    """สร้างคำตอบสุดท้ายจากเอกสารที่เกี่ยวข้อง"""
    llm = get_llm()
    # print(non_related_docs)

    if not related_docs:
        non_related_info = extract_non_related_info(non_related_docs)
        return generate_fallback_answer(query, non_related_info, llm)

    
    # รวมเนื้อหาจากเอกสารที่เกี่ยวข้อง
    content_parts = []
    resources = []
    
    for i, doc in enumerate(related_docs, 1):
        content_parts.append(f"ข้อมูลที่ {i}: {doc.page_content}")
        
        # รวบรวม resources
        if doc.metadata.get("urlfilenew"):
            resources.append(f"📄 {doc.metadata.get('topic', 'pdf')}: {doc.metadata.get('urlfilenew')}")
        if doc.metadata.get("clip"):
            resources.append(f"🎥 {doc.metadata.get('topic', 'video')}: {doc.metadata.get('clip')}")

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
ถ้าไม่เจอคำตอบในเนื้อหา ให้ตอบว่า "ผมไม่มั่นใจว่า นี่ใช่เอกสารที่คุณต้องการค้นหารึเปล่า ถ้าไม่ใช่หรือยังแก้ปัญหาไม่ได้ คุณสามารถติดต่อฝ่าย ICT ได้ที่เบอร์ 601(ระยอง) หรือ 116(กรุงเทพ) เพื่อขอความช่วยเหลือเพิ่มเติม"
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
            'linkpdf': pdf_link if pdf_link else "",
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
        typing_llm = Ollama(
            model= llm_model,
            base_url=base_url,
            temperature=0.1,
            num_predict=8,
            
        )
        type_query = typing_llm(define_intension_user_prompt + query + "\nคำตอบ ประเภทที่:")
        type_query = type_query.strip()
        import re
        numbers = re.findall(r'\d+', type_query)
        if numbers:
            type_query = int(numbers[0])
        else:
            type_query = 3  # Default to general question if no number found
    except Exception as e:
        print(e)
        type_query = 3  # ถ้าเกิด error ให้ถือว่าเป็นคำถามทั่วไป

    if type_query == 1:
        talking_llm = Ollama(
            model= llm_model,
            base_url=base_url,
            temperature=0.8,
            num_predict=300,
            num_ctx=10000,
        )
        final_answer = talking_llm(talking_prompt + query)
        return final_answer, ''
    elif type_query == 2:
        # 🔥 Enhanced phone search for NCR directory
        print("🔍 ค้นหาข้อมูลการติดต่อใน NCR Directory...")
        results = enhanced_phone_search(query, k=1)
        print(f"✅ พบข้อมูลการติดต่อ {len(results)} รายการ")
        
        # Use enhanced phone answer generation
        final_answer = generate_phone_answer(query, results)
        
        print("\n" + "="*60)
        print("📞 ข้อมูลการติดต่อ NCR:")
        print(final_answer)
        print("="*60)
        
        return final_answer, ''
    else:
        try:
            search =''
            need_all = typing_llm(define_user_need_in_rag+query+"คำคอบ: ")
            numbers = re.findall(r'\d+', need_all)
            if numbers[0] == '1':
                if 'p2a' in query.lower():
                    search= 'p2a' 
                elif 'ifaq' in query.lower():
                    search= 'ifaq'
                elif 'ax4' in query.lower():
                    search= 'ax4'
                elif 'rules' in query.lower():
                    search= 'rules'
                elif 'tip' in query.lower():
                    search= 'Tip dd by tc'
                else:
                    search = ''
                related_docs = vector_search(query, k=10,mode = 1,search=search)
                results =[{},{},{},{}]
                print("🔄 เริ่มค้นหาในฐานข้อมูล...")
            if search =='':
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
            final_answer = generate_final_answer(query, related_docs,results[0:3])
            
            print("\n" + "="*60)
            print("🤖 i-Bot:")
            print(final_answer)
            print("="*60)
            
            return final_answer , all_link
            
        except Exception as e:
            error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
            print(error_msg)
            return error_msg , ''


if __name__ == "__main__":
    text = ""
    while text != 'y':
        text = input("🔄 กด Enter เพื่อเริ่มการค้นหา หรือพิมพ์ 'y' เพื่อออก: ")
        print(process_query(text))
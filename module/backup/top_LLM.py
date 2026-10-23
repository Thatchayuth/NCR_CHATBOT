from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
# from phone_search_exact import search_phone
from phone_search_partial import search_phone
from find_com_excel import search_po_json
_embedding = None
_vectordb = None
_vectordb_phone = None
_llm = None
persist_directory = "last_persist_db_qwen_ollama_rev3"
persist_directory_phone = "phone_directory_vectordb"
all_link = []
#llm_model ="scb10x/llama3.1-typhoon2-8b-instruct:latest"
llm_model = "scb10x/typhoon2.5-qwen3-4b:latest"
# llm_model = "scb10x/typhoon2.1-gemma3-12b:latest"
base_url="http://10.10.1.10:11434"

define_intension_user_prompt = """วิเคราะห์และจัดประเภทคำถาม ตอบเฉพาะตัวเลข 1, 2, 3, หรือ 4:

ประเภท 1: การสนทนาทั่วไป
- คำทักทาย: "สวัสดี", "หวัดดี", "ดีครับ"
- คำขอบคุณ: "ขอบคุณ", "ขอบใจ", "อารีกาโต้"
- คำถามทั่วไป: "ทำอะไรได้บ้าง", "ช่วยอะไรได้", "เป็นยังไงบ้าง"
- การสนทนา: "วันนี้เป็นไง", "อากาศร้อน", "งานยุ่งไหม"

ประเภท 2: ค้นหาเบอร์โทร/ข้อมูลติดต่อ/อีเมลล์/หาemail
- เบอร์บุคคล: "เบอร์ของคุณสมชาย", "เบอร์พี่ก้อง", "โทรหาใคร"
- เบอร์แผนก: "เบอร์ฝ่ายบุคคล", "เบอร์ธุรการ", "เบอร์ ICT"
- เบอร์สถานที่: "เบอร์ระยอง", "เบอร์กรุงเทพ", "เบอร์สาขา"
- คำสำคัญ: "เบอร์", "โทร", "ติดต่อ", "หมายเลข", "โทรศัพท์" 
- คำถามตัวอย่าง: "ขอเบอร์โทรของคุณทองสร้อย", "หาเบอร์ของพี่ AW", "ขอเบอร์ฝ่ายธุรการ", "เบอร์ผู้จัดการ", "ขอเบอร์สาขาระยอง", "วิธีติดตั้ง Windows", "แก้ปัญหา Outlook ไม่เปิด", "ขอเบอร์ SPS", "สถานีตำรวจ", "ขอเบอร์พี่ก้องกับพี่แอน", "เบอร์คุณสมชายและคุณสมหญิง", "หาเบอร์ ICT กับ ธุรการ", "ขอเบอร์พี่ต้นแล้วก็พี่ใหม่", "เบอร์พี่ก้อง พี่แอน พี่ต้น"

ประเภท 3: คำถามเทคนิค/เอกสาร/ปฏิทิน
- ปัญหาคอมพิวเตอร์: "คอมช้า", "เปิดไม่ได้", "หน้าจอดับ"
- ซอฟต์แวร์: "Outlook ขัดข้อง", "Excel error", "Windows update"
- เอกสาร: "P2A", "i-FAQ", "AX4", "Tip dd by TC", "ICT rule"
- วิธีการ: "วิธีติดตั้ง", "วิธีแก้", "ทำยังไง", "Setting"
- วันหยุด , Core Value , WeCan

ประเภท 4: ข้อมูลเครื่องคอมพิวเตอร์
- รหัสเครื่อง: "รหัสเครื่องของ", "หมายเลขเครื่อง", "NBK", "NCR"
- อายุเครื่อง: "อายุเครื่อง", "เครื่องเก่าแค่ไหน", "ใช้มานานไหม"
- ข้อมูลเครื่อง: "ชื่อเครื่อง", "สเปค", "รายละเอียดเครื่อง"
- ตัวอย่าง: "อายุเครื่อง NBK649", "รหัสเครื่องของสมชาย"



ตอบเฉพาะตัวเลข 1, 2, 3, หรือ 4
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

define_user_need_in_rag = """วิเคราะห์ประเภทคำถามและแยกความต้องการของผู้ใช้:

🎯 ประเภทที่ 1: ต้องการสารบัญ/รายการทั้งหมด
- คำถามที่มีคำว่า: "ทั้งหมด", "สารบัญ", "รายการ", "หัวข้อ", "มีอะไรบ้าง", "ประเภทไหน", "แนะนำ"
- ตัวอย่าง: "P2A มีอะไรบ้าง", "สารบัญ i-FAQ", "หัวข้อทั้งหมด", "แนะนำหัวข้อ", "มีเอกสารอะไร"

🎯 ประเภทที่ 2: ต้องการข้อมูลเฉพาะเจาะจง
- คำถามที่ถามเรื่องเฉพาะ หรือปัญหาใดปัญหาหนึ่ง
- ตัวอย่าง: "แก้ Outlook ไม่เปิด", "ติดตั้ง Windows", "เปลี่ยนรหัส wifi", "ปริ้นเตอร์ขัดข้อง"

กฎการตัดสิน:
✅ ถ้าคำถามถามถึงรายการ/สารบัญ/ทั้งหมด → ตอบ 1
✅ ถ้าคำถามถามเรื่องเฉพาะ/ปัญหาใดปัญหาหนึ่ง → ตอบ 2
✅ กรณีสงสัย: ถ้าไม่แน่ใจให้เลือก 2

⚠️ ตอบเฉพาะตัวเลข 1 หรือ 2 เท่านั้น ห้ามอธิบาย

คำถาม: """

define_user_input = """วิเคราะห์คำถามของผู้ใช้และแยกเอาคำค้นหาที่แท้จริงออกมา

กฎการแยก:

1. ถ้าคำถามเป็นการขอเบอร์โทร ให้แยกเอาชื่อคน/แผนก/ตำแหน่งออกมา
2. ตัดคำที่ไม่จำเป็นออก เช่น "ขอ", "เบอร์", "โทร", "ของ", "คุณ", "พี่", "น้อง"


ตัวอย่าง:
- "ขอเบอร์โทรของคุณทองสร้อย" → ทองสร้อย
- "หาเบอร์ของพี่ AW" → AW  
- "ขอเบอร์ฝ่ายธุรการ" → ฝ่ายธุรการ
- "เบอร์ผู้จัดการ" → ผู้จัดการ
- "ขอเบอร์สาขาระยอง" → ระยอง
- "วิธีติดตั้ง Windows" c ติดตั้ง Windows
- "แก้ปัญหา Outlook ไม่เปิด" → Outlook ไม่เปิด
- "ขอเบอร์ SPS" → SPS
- "สถานีตำรวจ" → ตำรวจ
- "ขอเบอร์พี่ก้องกับพี่แอน" → ก้อง|แอน
- "เบอร์คุณสมชายและคุณสมหญิง" → สมชาย|สมหญิง
- "หาเบอร์ ICT กับ ธุรการ" → ICT|ธุรการ
- "ขอเบอร์พี่ต้นแล้วก็พี่ใหม่" → ต้น|ใหม่
- "เบอร์พี่ก้อง พี่แอน พี่ต้น" → ก้อง|แอน|ต้น

กฎเพิ่มเติม:
- ถ้ามีหลายชื่อหรือหลายคำค้นหา ให้คั่นด้วยเครื่องหมาย | (pipe)
- ถ้ามีคำว่า "กับ", "และ", "แล้วก็", "with", "and" แสดงว่ามีหลายคำค้นหา ให้แยกด้วย |

ข้อห้าม:
- ห้ามเปลี่ยนคำ จาก SPS เป็น SPES แบบนี้ผิดกฎนะครับ
- ห้ามแปลความหมาย
- ห้ามใช้คำทับศัพท์
คำถาม: """

define_user_input_com = """วิเคราะห์คำถามของผู้ใช้และแยกเอาคำค้นหาที่แท้จริงออกมา


กฎการแยก:

1. ตัดคำที่ไม่จำเป็นออก เช่น "ขอ", "เบอร์", "โทร", "ของ", "คุณ", "พี่", "น้อง"
2. ตอบแค่คำค้นหาที่สำคัญเท่านั้น ไม่ต้องอธิบาย

ตัวอย่าง:
- อยากรู้อายุเครื่องของ  NBK649 → NBK649
- อยากรู้รหัสเครื่่องของ Natapong → Natapong
- อยากรู้หมายเลขเครื่องของ AW → AW

คำถาม: """

def get_llm():
    global _llm
    if _llm is None:
        print("🔄 โหลด LLM ครั้งแรก...")
        _llm = OllamaLLM(
            model= llm_model,
            # model = "qwen3:0.6b",
            # model = "llama3:8b",
            # model = "llama3.1:8b",
            base_url="http://10.10.1.10:11434",
            temperature=0.1,
            top_p=0.3,
            top_k=60, 
            num_predict= 3000,
            num_ctx = 50000,
            # base_url="https://89kpfn5l-11434.asse.devtunnels.ms"
        )
    return _llm

def get_embedding():
    global _embedding
    if _embedding is None:
        print("🔄 โหลด embedding model ครั้งแรก...")
        _embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device": "cpu"})
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
    elif mode == 1 :
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
📞 เบอร์ 601 (ระยอง) หรือ 145 (กรุงเทพ) นะครับ"""
    
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
        answer = llm.invoke(phone_answer_prompt)
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
- calendar (ถ้าต้องการปฏิทิน)
- ไม่ต้องการ (ถ้าแค่ต้องการคำอธิบาย/ข้อมูล)

คำตอบ:"""
    user_need = llm.invoke(check_need_template)
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
    clean_query = llm.invoke(f"""ทำความสะอาดคำถามนี้ให้กระชับและเข้าใจง่ายโดยพยายามตัดพวก PDF หรีอ VDO ออก คำถาม: "{query} """)
    print(clean_query)
    # ✅ แก้ไข: ใช้ template parameter
    prompt_template = PromptTemplate(
        input_variables=["question", "topic", "content_preview"],
        template="""ตรวจสอบความเกี่ยวข้องระหว่างคำถามและเนื้อหา:
    
        คำถาม: {question}

        หัวข้อ: {topic}
        เนื้อหา: {content_preview}

        ตอบแค่ YES หรือ NO หรือ MAY BE:
        - YES = เนื้อหาและหัวข้อ นี้ตอบคำถามได้หรือเกี่ยวข้องกับคำถาม
        - NO = เนื้อหาและหัวข้อนี้ไม่เกี่ยวข้องกับคำถาม
        - MAY BE = เนื้อหา หรือ หัวข้อ อาจจะเกี่ยวข้องกับคำถาม

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
            
            response = llm.invoke(prompt)
            print(response)
            response = response.strip().lower()
            
            if "yes" in response or "may be" in response.lower():
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
        "📞 เบอร์ 601 (ระยอง) หรือ 145 (กรุงเทพ)"
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
        llm_creative = OllamaLLM(
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
            base_url="http://127.0.0.1:11434",
            temperature=0.3,  # เพิ่มความเป็นธรรมชาติ
            top_p=0.8,
            top_k=200,
            num_predict=600,
            num_ctx=8000,
        )
        improved_answer = llm_creative.invoke(improvement_prompt)
        return improved_answer.strip()
    except Exception as e:
        print(f"⚠️ Error improving fallback answer: {e}")
        return raw_answer

def clean_search_input(input_text: str) -> str:
    """
    Clean the input to only allow letters and numbers
    
    Args:
        input_text: Raw input from LLM
        
    Returns:
        Cleaned string with only letters and numbers
    """
    import re
    
    # Remove quotes and extra whitespace
    cleaned = input_text.strip().strip('"\'')
    
    # Keep only letters (Thai and English) and numbers
    # \u0E00-\u0E7F is Thai character range
    cleaned = re.sub(r'[^\w\u0E00-\u0E7F0-9\s]', '', cleaned)
    
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

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
        print('#'*50)
        print(doc.page_content)

        # รวบรวม resources
        if doc.metadata.get("urlfilenew"):
            resources.append(f"📄 {doc.metadata.get('topic', 'pdf')}: {doc.metadata.get('urlfilenew')}")
        if doc.metadata.get("clip"):
            resources.append(f"🎥 {doc.metadata.get('topic', 'video')}: {doc.metadata.get('clip')}")
        if doc.metadata.get("picture"):
            resources.append(f"🎥 {doc.metadata.get('topic', 'picture')}: {doc.metadata.get('picture')}")

    combined_content = "\n\n".join(content_parts)
    # print(combined_content)
    
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
ถ้าไม่เจอคำตอบในเนื้อหา ให้ตอบว่า "ผมไม่มั่นใจว่า นี่ใช่เอกสารที่คุณต้องการค้นหารึเปล่า ถ้าไม่ใช่หรือยังแก้ปัญหาไม่ได้ คุณสามารถติดต่อฝ่าย ICT ได้ที่เบอร์ 601(ระยอง) หรือ 145(กรุงเทพ) เพื่อขอความช่วยเหลือเพิ่มเติม"
คำตอบ:"""
    
    try:
        
        answer = llm.invoke(answer_prompt)
        
        # เพิ่ม resources ถ้ามี
        if resources:
            answer += "\n\n📎 ลิงก์ที่เกี่ยวข้อง:\n" + "\n".join(resources)
            
        return answer.strip()
        
    except Exception as e:
        return f"ขอโทษครับ เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"

def generate_final_answer_json(query,json_data):
    llm = get_llm()
    answer_prompt = f"""คุณคือ i-Bot เด็กชายอายุ 18 ปี แผนก ICT บริษัท NCR Rubber
        คุณเป็นผู้ช่วยด้าน IT ที่เป็นมิตรและให้ความช่วยเหลือ

        กฎการตอบ:
        1. ใช้ "ครับ" และ "นะครับ" ในทุกประโยค
        2. ตอบให้กระชับ เข้าใจง่าย
        3. ใช้ข้อมูลจากเนื้อหาที่ให้มาในการตอบ
        4. หากมีขั้นตอนการแก้ไข ให้เรียงลำดับชัดเจน

        Client หมายถึง หมายเลขเครื่อง
        User หมายถึง ชื่อผู้ใช้
        PO_Date หมายถึง วันที่ซื้อเครื่อง
        
        คำถาม: {query}

        เนื้อหาที่เกี่ยวข้อง:
        {json_data}
        ถ้าไม่เจอคำตอบในเนื้อหา ให้ตอบว่า "ผมไม่มั่นใจว่า นี่ใช่เครื่องที่คุณต้องการค้นหารึเปล่า ถ้าไม่ใช่หรือยังแก้ปัญหาไม่ได้ คุณสามารถติดต่อฝ่าย ICT ได้ที่เบอร์ 601(ระยอง) หรือ 145(กรุงเทพ) เพื่อขอความช่วยเหลือเพิ่มเติม"
        คำตอบ:"""
    try:
        answer = llm.invoke(answer_prompt)
        

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
    pic_link = None
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
        
        if not pic_link and doc.metadata.get('picture'):
            pic_link = doc.metadata.get('picture')
            print(f"   🎥 เลือก Video: {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
        # ถ้าได้ครบทั้งสองแล้ว ก็หยุด
        if pdf_link and video_link:
            break
    
    # สร้าง link object
    if pdf_link or video_link or pic_link:
        link = {
            'link_pdf': pdf_link if pdf_link else "",
            'link_clip': video_link if video_link else "",
            'link_pic':pic_link if pic_link else ""
        }
        all_link.append(link)
        print(f"   🔗 เพิ่มลิงก์: PDF={'มี' if pdf_link else 'ไม่มี'}, Video={'มี' if video_link else 'ไม่มี'}")
    
    return pdf_link, video_link

# def process_query(query,db_mode=4):

def process_query(query):
    global all_link
    """ฟังก์ชันประมวลผลคำถาม"""
    if not query.strip():
        print("⚠️ กรุณาใส่คำถามครับ")
        return
    
    try:
        typing_llm = OllamaLLM(
            model= llm_model,
            base_url=base_url,
            temperature=0.1,
            num_predict=50,
            
        )
        if 'we can' in query.lower():
            type_query = 3
        else:
            type_query = typing_llm.invoke(define_intension_user_prompt + query +'ตอบเฉพาะตัวเลข 1, 2, 3, หรือ 4'+ "\nคำตอบ ประเภทที่:")
            type_query = type_query.strip()
        import re
        numbers = re.findall(r'\d+', type_query)
        print(f"🔍 ประเภทคำถามที่วิเคราะห์ได้: {type_query}")
        print('################################')
        if numbers:
            type_query = int(numbers[0])
        else:
            type_query = 3  # Default to general question if no number found
    except Exception as e:
        print(e)
        type_query = 3  # ถ้าเกิด error ให้ถือว่าเป็นคำถามทั่วไป
    # if db_mode != type_query:
    #     # Update user mode in the database
    #     if type_query == 2:
    #         db_mode = 2
    #     return f"🔄 ต้องการเปลี่ยนค้นหา"
    if type_query == 1:
        talking_llm = OllamaLLM(
            model= llm_model,
            base_url=base_url,
            temperature=0.8,
            num_predict=300,
            num_ctx=10000,
        )
        final_answer = talking_llm.invoke(talking_prompt + query)
        mode = 1
        return final_answer, '',mode
    elif type_query == 2:
        # 🔥 Enhanced phone search for NCR directory
        # print("🔍 ค้นหาข้อมูลการติดต่อใน NCR Directory...")
        # results = enhanced_phone_search(query, k=1)
        # print(f"✅ พบข้อมูลการติดต่อ {len(results)} รายการ")
        
        # # Use enhanced phone answer generation
        # final_answer = generate_phone_answer(query, results)
        
        # print("\n" + "="*60)
        # print("📞 ข้อมูลการติดต่อ NCR:")
        # print(final_answer)
        # print("="*60)
        ignore_word_list = ['ด่วน','สถานี','email','e-mail','อีเมล','อีเมลล์','Email','E-mail']
        for word in ignore_word_list:
            query=query.replace(word,'')
        # query = query.replace('ด่วน','')
        # query = query.replace('สถานี','')
        # query = query.replace('email','')
        # query = query.replace('e-mail','')
        # query = query.replace('อีเมล','')
        # query = query.replace('อีเมลล์','')
        clean_input = typing_llm.invoke(define_user_input + query + "\nคำค้นหาคือ:")
        # clean_input = typing_llm(define_user_input_com + query + "\nคำค้นหา:")
        clean_input = clean_input.strip()
        
        print(f"🔍 คำถามเดิม: {query}")
        print(f"🎯 คำค้นหาที่แยกได้: {clean_input}")

        # รองรับการค้นหาหลายคน (คั่นด้วย |)
        if '|' in clean_input:
            search_terms = [term.strip() for term in clean_input.split('|') if term.strip()]
            print(f"📋 พบคำค้นหา {len(search_terms)} รายการ: {search_terms}")
            
            all_results = []
            for term in search_terms:
                print(f"🔍 ค้นหา: {term}")
                result = search_phone(term)
                if result:
                    all_results.extend(result)
            
            final_answer = all_results if all_results else []
        else:
            final_answer = search_phone(clean_input)

        mode = 2
        return final_answer, '',mode
    elif type_query == 4:
        clean_input = typing_llm.invoke(define_user_input_com + query + "\nคำค้นหา:")
        clean_input = clean_search_input(clean_input)
        print(f"🔍 คำค้นหาเครื่อง: {clean_input}")
        data = search_po_json(clean_input)
        if data:
            final_answer = generate_final_answer_json(query, data)
            mode = 4
            return final_answer, '', mode
        else:
            mode = 4
            return "ขอโทษครับ ไม่พบข้อมูลเครื่องคอมพิวเตอร์ที่ค้นหาครับ", '', mode
    else:
        try:
            mode = 3
            search =''
            need_all = typing_llm.invoke(define_user_need_in_rag+query+"คำคอบ: ")
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
                results = vector_search(query, k=8)
                print(f"✅ พบเอกสาร {len(results)} รายการจากการค้นหา")
                
                print("\n🤖 กำลังใช้ LLM กรองข้อมูล...")
                related_docs = match_query_with_vector_db(query, results)
                print(f"🎯 พบเอกสารที่เกี่ยวข้อง {len(related_docs)} รายการ")
                
                related_docs = remove_duplicate_resources(related_docs)
                
                if related_docs:
                    count_type = 0
                    print("\n📋 รายการเอกสารที่เกี่ยวข้อง:")
                    all_topic=[]
                    for i, doc in enumerate(related_docs, 1):
                        if doc.metadata.get('type') =='Core Value':
                            count_type += 1
                        all_topic.append(doc.metadata.get('topic', 'ไม่มีหัวข้อ'))
                        print(f"   {i}. {doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
                    if count_type > 4:
                        topic_number = 1
                        text_all_topic = ""
                        for topic in all_topic:
                            text_all_topic += f"{topic_number}: {topic}\n"
                            topic_number += 1
                        #จัดเรียงเอกสารใน related_docs ใหม่
                        sorted_promt = "จากหัวข้อต่อไปนี้\n" + text_all_topic + " จงเลือกหัวข้อที่ตรงกับคำถามมากที่สุด\n" +'คำถามคือ:\n'+ query + "\nให้ตอบเป็นตัวเลขเท่่านั้น"
                        topic_choice = typing_llm.invoke(sorted_promt + "\nคำตอบคือเลข: ")
                        print('LLMเลือก:')
                        print(topic_choice)
                        print('#'*50)
                        topic_choice = topic_choice.strip()
                        
                        numbers = re.findall(r'\d+', topic_choice)
                        if numbers:
                            choice_index = int(numbers[0]) - 1
                            if 0 <= choice_index < len(related_docs):
                                chosen_doc = related_docs[choice_index]
                                related_docs = [chosen_doc] + [doc for i, doc in enumerate(related_docs) if i != choice_index]
                                print(f"🔍 เลือกหัวข้อที่ตรงกับคำถาม: {chosen_doc.metadata.get('topic', 'ไม่มีหัวข้อ')}")
                pdf_link, video_link = collect_unique_links(related_docs)
            
                print("\n🤖 กำลังสร้างคำตอบ...")

            final_answer = generate_final_answer(query, related_docs,results[0:3])
            print("\n" + "="*60)
            print("🤖 i-Bot:")
            print(final_answer)
            print("="*60)
            
            return final_answer , all_link,mode
            
        except Exception as e:
            error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
            print(error_msg)
            return error_msg , '',mode


if __name__ == "__main__":
    text = ""
    while text != 'y':
        text = input("🔄 กด Enter เพื่อเริ่มการค้นหา หรือพิมพ์ 'y' เพื่อออก: ")
        print(process_query(text))
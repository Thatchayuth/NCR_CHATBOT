from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
# from langchain.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import requests
from typing import Optional, List
from collections import defaultdict
from langchain.document_loaders import PyMuPDFLoader
import tempfile
from pdf_video_analyzer_2 import transcribe_video_url_with_transformers,extract_text_from_pdf
import sqlite3
import json
import pyodbc
import threading
from datetime import datetime, timedelta
import pandas as pd

_embedding = None
_vectordb = None
_llm = None

class CustomOllamaLLM(LLM):
    model: str
    base_url: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"].strip()

    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

llm = Ollama(
    model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
    # model="qwen3:1.7b",
    # model = "qwen3:8b",
    # model = "deepseek-r1:8b",
    # model = "llama3.1:8b",
    # model = "llama3:8b",
    base_url="http://127.0.0.1:11434"
)

# print("✅ LLM พร้อมใช้งาน")
# print("🔧 กำลังโหลด Vector DB...")
# embedding = OllamaEmbeddings(
#     model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
#     base_url="http://127.0.0.1:11434"
# )
persist_directory = "last_persist_db_qwen_ollama_rev2"
# vectordb = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embedding
# )
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# retriever = vectordb.as_retriever()
# search = DuckDuckGoSearchRun()

# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,return_source_documents=True)
# def company_info_tool(input_text):
#     result = qa_chain.invoke(input_text)
#     return result['result']
user_memories = defaultdict(lambda: ConversationBufferMemory(memory_key="chat_history", return_messages=True))

# SQL Server Database Configuration
DB_CONFIG = {
    'server': 'localhost\\SQLEXPRESS',
    'database': 'Cedar6_ICT', 
    'username': 'sa',
    'password': 'top38759'
}


def get_sql_connection():
    """Get SQL Server database connection"""
    connection_string = f'''
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={DB_CONFIG['server']};
    DATABASE={DB_CONFIG['database']};
    UID={DB_CONFIG['username']};
    PWD={DB_CONFIG['password']};
    TrustServerCertificate=yes;
    '''
    return pyodbc.connect(connection_string)

def search_sql_database(search_value):
    """
    Search for a specific value across all tables and columns in SQL Server database
    Returns JSON formatted results
    """
    search_value = search_value.strip()
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        
        # Get all tables and their columns
        cursor.execute("""
            SELECT 
                TABLE_NAME, 
                COLUMN_NAME, 
                DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME IN (
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
            )
            ORDER BY TABLE_NAME, COLUMN_NAME
        """)
        
        columns_info = cursor.fetchall()
        found_results = []
        
        for table_name, column_name, data_type in columns_info:
            # Search in string/text type columns
            if data_type.lower() in ['varchar', 'nvarchar', 'char', 'nchar', 'text', 'ntext']:
                try:
                    # Search for the value (case-insensitive)
                    search_query = f"""
                        SELECT TOP 5 *
                        FROM [{table_name}] 
                        WHERE LOWER(CAST([{column_name}] AS NVARCHAR(MAX))) LIKE LOWER('%{search_value}%')
                    """
                    
                    cursor.execute(search_query)
                    records = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    
                    for record in records:
                        # Convert record to dictionary
                        record_dict = {}
                        for i, value in enumerate(record):
                            record_dict[column_names[i]] = str(value) if value is not None else None
                        
                        result_item = {
                            'table': table_name,
                            'found_in_column': column_name,
                            'record_data': record_dict
                        }
                        found_results.append(result_item)
                    
                except Exception:
                    continue
        
        conn.close()
        
        if found_results:
            return json.dumps({
                'status': 'success',
                'search_value': search_value,
                'total_found': len(found_results),
                'results': found_results
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                'status': 'not_found',
                'search_value': search_value,
                'message': f'ไม่พบข้อมูล "{search_value}" ในฐานข้อมูล'
            }, ensure_ascii=False, indent=2)
            
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'search_value': search_value,
            'error': f'เกิดข้อผิดพลาด: {str(e)}'
        }, ensure_ascii=False, indent=2)

def execute_sql_query(query):
    """
    Execute a custom SQL query (SELECT only for safety)
    Returns JSON formatted results
    """
    query = query.strip()
    
    # Safety check - only allow SELECT queries
    if not query.upper().startswith('SELECT'):
        return json.dumps({
            'status': 'error',
            'error': 'เพื่อความปลอดภัย อนุญาตเฉพาะ SELECT query เท่านั้น'
        }, ensure_ascii=False, indent=2)
    
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        
        cursor.execute(query)
        records = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        # Convert results to list of dictionaries
        results = []
        for record in records:
            record_dict = {}
            for i, value in enumerate(record):
                record_dict[column_names[i]] = str(value) if value is not None else None
            results.append(record_dict)
        
        conn.close()
        
        return json.dumps({
            'status': 'success',
            'query': query,
            'total_rows': len(results),
            'columns': column_names,
            'results': results
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'query': query,
            'error': f'เกิดข้อผิดพลาดในการรัน query: {str(e)}'
        }, ensure_ascii=False, indent=2)

def get_database_tables():
    """
    Get list of all tables in the database with row counts
    """
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            t.TABLE_NAME,
            ISNULL(p.rows, 0) as ROW_COUNT
        FROM INFORMATION_SCHEMA.TABLES t
        LEFT JOIN (
            SELECT 
                OBJECT_NAME(object_id) as table_name,
                SUM(rows) as rows
            FROM sys.partitions 
            WHERE index_id IN (0,1)
            GROUP BY object_id
        ) p ON t.TABLE_NAME = p.table_name
        WHERE t.TABLE_TYPE = 'BASE TABLE'
        ORDER BY t.TABLE_NAME
        """
        
        cursor.execute(query)
        records = cursor.fetchall()
        
        tables = []
        for record in records:
            tables.append({
                'table_name': record[0],
                'row_count': record[1]
            })
        
        conn.close()
        
        return json.dumps({
            'status': 'success',
            'database': DB_CONFIG['database'],
            'total_tables': len(tables),
            'tables': tables
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'error': f'เกิดข้อผิดพลาด: {str(e)}'
        }, ensure_ascii=False, indent=2)

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

def get_llm():
    global _llm
    if _llm is None:
        print("🔄 โหลด LLM ครั้งแรก...")
        _llm = CustomOllamaLLM(
            model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
            # model = "llama3:8b",
            base_url="http://127.0.0.1:11434"
        )
    return _llm

def i_bot_answer(question):
    try:
        # embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device": "cuda"})
        # # vectordb = Chroma(
        # #     persist_directory=persist_directory,
        # #     embedding_function=embedding
        # # )

        # embedding = get_embedding()
        vectordb = get_vectordb()
        llm = get_llm()
        TOP_K = 1
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        # llm = CustomOllamaLLM(
        #     model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        #     # model="typhoon",
        #     base_url="http://127.0.0.1:11434"
        # )
        
    
        # custom_prompt = PromptTemplate.from_template("""
        #     คุณ คือ "i-Bot" รับหน้าที่พนักงานITชายของบริษัท NCR Rubber
        #     หากคำถามเกินขอบเขตข้อมูลที่คุณรู้ คุณจะตอบด้วยความสุภาพว่ายังไม่มีข้อมูล พร้อมแนะนำว่าข้อมูลที่คุณมีอยู่มีอะไรบ้าง

        #     ภารกิจ: เปรียบเทียบหัวข้อ ให้สั้นที่สุด 
        #     - ตอบคำถามให้กระชับและเข้าใจง่าย เหมือนเล่าให้เพื่อนฟัง
        #     - เขียนประโยคธรรมชาติ ไม่เป็นทางการเกินไป
            

        #     เอกสารอ้างอิง:
        #     {context}

        #     คำถาม: {question}                                          
        #     """)
        custom_prompt = PromptTemplate.from_template("""
            ตอบคำถามให้เป็นภาษาไทย เท่านั้น
            นายคือ ibot Chatbot พนักงานเด็กชายอายุ 18 แผนก ICT ของบริษัท NCR Rubber
            ภารกิจ: ตอบคำถามให้กระชับและเข้าใจง่าย เหมือนเล่าให้เพื่อนฟัง
            - เขียนประโยคธรรมชาติ ไม่เป็นทางการเกินไป
            - ใช้เอกสารอ้างอิงที่เกี่ยวข้องเพื่อช่วยในการตอบคำถาม

            เอกสารอ้างอิง:
            {context}

            คำถาม: {question}
        """)

        results = vectordb.similarity_search_with_score(question, k=3)
        #print(results)
        

        confidents = []
        for confident in results:
            print("🔍 ความมั่นใจ:", confident[1])
            confidents.append(confident[1])

        qa_chain = RetrievalQA.from_chain_type(
                llm        = llm,
                retriever  = retriever,
                return_source_documents = True,
                chain_type_kwargs = {"prompt": custom_prompt}
            )
        result = qa_chain.invoke({"query": question})
        link_pdf =[]
        link_clip =[]
        all_link =[]
        for doc in result["source_documents"]:
            print(doc)
            print('#**10')
            print("📄", doc.page_content)
            url_pdf = doc.metadata.get("urlfilenew", "")
            url_clip = doc.metadata.get("clip", "")
            # link_pdf =[]
            # link_clip =[]
            # if url_pdf:
            #     link_pdf.append(url_pdf)
            # if url_clip:
            #     link_clip.append(url_clip)
            # if url_pdf:
            #     if url_pdf[-3] != 'pdf':
            #         url_pdf= url_pdf+['.pdf']
            data = {
                "link_pdf":url_pdf,
                "link_clip":url_clip
            }
            all_link.append(data)
        all_data ={
            # 'answer': {'result':result['result'],'link':all_link},
            'answer':result['result'],
            'link': all_link
        }
        # return json.dumps(all_data, ensure_ascii=False, indent=2)
        return all_data
    except Exception as e:
        return {
            'error': f"เกิดข้อผิดพลาดในการประมวลผลคำถาม: {str(e)}"
        }

def this_is_a_book(question):
    return 'this is a book'

def summarize_pdf(file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # ถ้าเป็น URL ให้ดาวน์โหลด PDF ก่อน
        print(f"📥 กำลังดาวน์โหลด PDF จาก {file_path}...")
        text = get_text_from_url_pdf(file_path)
    else:
        text = extract_text_from_pdf(file_path)
    
    print(text)
    if not text or len(text.strip()) == 0:
        return "ไม่พบเนื้อหาในไฟล์ PDF"

    # ตัดแบ่ง text เป็น chunk ย่อยๆ (เช่น 1500 ตัวอักษร, overlap 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # สรุปแต่ละ chunk แล้วรวมผลลัพธ์
    summaries = []
    for chunk in chunks:
        prompt = f"สรุปเนื้อหาต่อไปนี้ให้กระชับและเข้าใจง่ายเป็นภาษาไทย:\n{chunk}"
        summary = llm(prompt)
        summaries.append(summary)

    # รวมสรุปทั้งหมดเป็นข้อความเดียว
    return "\n\n".join(summaries)

def find_computer_with_maintenance(pc_number):
    pc_number =pc_number.strip().upper()  # ทำให้เป็นตัวพิมพ์ใหญ่และตัดช่องว่าง
    try:
        with sqlite3.connect('computer_inventory.db') as conn:
            conn.row_factory = sqlite3.Row  # This allows dict-like access
            cursor = conn.cursor()
            
            # Get computer info
            cursor.execute("SELECT * FROM computers WHERE computer_id = ?", (pc_number,))
            computer_row = cursor.fetchone()
            
            if not computer_row:
                return json.dumps({"error": f"ไม่พบข้อมูลสำหรับ {pc_number}"}, ensure_ascii=False, indent=2)
            
            # Convert computer data to dict
            computer_data = dict(computer_row)
            
            # Get maintenance logs
            cursor.execute("""
                SELECT * FROM maintenance_logs 
                WHERE computer_id = ? 
                ORDER BY maintenance_date DESC
            """, (pc_number,))
            maintenance_rows = cursor.fetchall()
            
            # Convert maintenance data to list of dicts
            maintenance_data = [dict(row) for row in maintenance_rows]
            
            # Combine all data
            result = {
                "computer_info": computer_data,
                "maintenance_history": maintenance_data,
                "total_maintenance_count": len(maintenance_data),
                "total_maintenance_cost": sum(row.get('cost', 0) or 0 for row in maintenance_data)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"เกิดข้อผิดพลาด: {str(e)}"}, ensure_ascii=False, indent=2)

def get_text_from_url_pdf(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่าดาวน์โหลดสำเร็จ
        print(f"📥 ดาวน์โหลด PDF สำเร็จจาก {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        return "\n".join(doc.page_content for doc in documents)

    except Exception as e:
        print(f"❌ Error reading PDF from URL: {e}")
        return ""

def get_agent_ans(user_id, question):
    """
    ฟังก์ชันสำหรับรับคำตอบจาก agent ตาม user_id และคำถามที่ระบุ
    """
    def release_my_chat_memory(text):
        """ล้างหน่วยความจำแชทของผู้ใช้ปัจจุบัน"""
        try:
            # ใช้ user_id ที่ส่งมาจาก closure แทนการให้ user พิมพ์
            if user_id in user_memories:
                user_memories[user_id].chat_memory.messages.clear()
                return f"✅ ล้างหน่วยความจำแชทของคุณเรียบร้อยแล้ว"
            else:
                return f"❌ ไม่พบหน่วยความจำสำหรับคุณ"
        except Exception as e:
            return f"❌ เกิดข้อผิดพลาด: {str(e)}"
    if ".pdf" in question.lower():
        tools = [
            Tool(
                name='ตัวช่วยอ่านและสรุปเอกสาร',
                func=summarize_pdf,
                description="ใช้สำหรับอ่านและสรุปเอกสาร PDF,หรือลิงค์ ที่ให้มา"
            )
        ]
    elif '.mp4' in question.lower():
        tools = [
            Tool(
                name='ถอดเสียงวิดีโอ',
                func=transcribe_video_url_with_transformers,
                description="ใช้สำหรับถอดเสียงวิดีโอจาก URL ที่ให้มาเป็นข้อความภาษาไทย"
            )
        ]
    else:
        tools = [
        Tool(
            name="RAG_Knowledge_Search",
            func=i_bot_answer,
            description="ใช้สำหรับตอบคำถามส่วนใหญ่,laptop,notebook, ทางด้านเทคนิค,zoom,ความรู้Excel , IT,ICT ,P2A, Printer, Email,วิธีการถนอมโน๊ตบุ๊ค,วิธีแก้ปัญหาต่างๆ เช่น คำถามเกี่ยวกับโน๊ตบุ๊ค,เครื่องพิมพ์,อีเมล,การใช้งานโปรแกรมต่างๆ"
        ),

        # Tool(
        #     name='ตัวช่วยอ่านและสรุปเอกสาร',
        #     func= summarize_pdf,
        #     description="ใช้สำหรับอ่านและสรุปเอกสาร PDF ที่ให้มา"
        # ),
        # Tool(
        #     name='Internal Phone Search',
        #     func= phone_search,
        #     description="ใช้สำหรับค้นหาหมายเลขโทรศัพท์ภายในบริษัท NCR Rubber จากชื่อย่อ หรือ ชื่อแผนก"
        # ),
        # Tool(
        #     name='ถอดเสียงวิดีโอ',
        #     func=transcribe_video_url_with_transformers,
        #     description="ใช้สำหรับถอดเสียงวิดีโอจาก URL ที่ให้มาเป็นข้อความภาษาไทย"),    
        
   #     Tool(
   #         name='computer maintanance database',
   #         func=find_computer_with_maintenance,
   #         description="ใช้สำหรับหาข้อมูลของโน๊ตบุ๊คแต่ละเครื่อง เช่น NCR230 , NBK118 โดยใช้ข้อมูลจาก  Database MySQl เป็นหลัก"
   #     ),
        Tool(
            name='ล้างหน่วยความจำแชท ลืมแชท',
            func=release_my_chat_memory,
            description="ใช้ลืมแชท,เริ่มแชทใหม่,ล้างแชท,เคลียแชท,ใช้สำหรับล้างหน่วยความจำแชทของผู้ใช้ "
        ),
        # Tool(
        #     name='ค้นหาข้อมูลในฐานข้อมูล',
        #     func=search_sql_database,
        #     description="ใช้สำหรับค้นหาข้อมูลในฐานข้อมูล SQL Server เช่น ค้นหาหมายเลขคอมพิวเตอร์ ชื่อผู้ใช้ หรือข้อมูลอื่นๆ เช่น NCR118 ,NBK105 "
        # ),
        # Tool(
        #     name='รัน SQL Query',
        #     func=execute_sql_query,
        #     description="ใช้สำหรับรัน SQL SELECT query เพื่อดึงข้อมูลเฉพาะจากฐานข้อมูล"
        # ),
        # Tool(
        #     name='ดูรายการตารางฐานข้อมูล',
        #     func=get_database_tables,
        #     description="ใช้สำหรับดูรายการตารางทั้งหมดในฐานข้อมูล Cedar6_ICT พร้อมจำนวนแถวข้อมูล"
        # )
        ]

    def get_agent_for_user(user_id, personality="friendly"):
        memory = user_memories[user_id]
        personalities = {
        "default": """คุณคือ 'i-Bot' พนักงานเด็กชายอายุ 18 แผนก  ICT ของบริษัท NCR Rubber ที่เป็นมิตรและช่วยเหลือผู้ใช้ เป็นผู้เชี่ยวชาญด้าน IT และสามารถตอบคำถามเกี่ยวกับเทคโนโลยีต่างๆ ได้อย่างรวดเร็วและแม่นยำ
        ในประโยคควรมีคำว่า*****ครับ,นะครับ*****

        1. หากเจอ URL ที่ลงท้ายด้วย .pdf หรือมีคำว่า "เอกสาร" "สรุป" ให้ใช้เครื่องมือสำหรับ PDF
        2. หากเจอ URL ที่ลงท้ายด้วย .mp4 หรือมีคำว่า "วิดีโอ" "ถอดเสียง" ให้ใช้เครื่องมือสำหรับวิดีโอ
        3. หากเจอรหัสคอมพิวเตอร์ (NCR, NBK + ตัวเลข) ให้ใช้เครื่องมือค้นหาข้อมูลคอมพิวเตอร์
        4. หากมีคำว่า "ล้าง" "ลืม" "เริ่มใหม่" ให้ใช้เครื่องมือล้างหน่วยความจำ
        5. สำหรับคำถามทั่วไปด้านเทคนิค ให้ใช้เครื่องมือRAG_Knowledge_Search **พยายามป้อมคำถามเป็นภาษาไทยให้มากที่สุด** **พยายามส่งลิงค์ที่เกี่ยวข้องกับคำถามให้มากที่สุด**
        6. คำถามเกี่ยวกับเบอร์โทร/ชื่อคน/แผนก → ใช้ "Internal Phone Search"
        7. ถ้าคำถามไม่เกี่ยวข้องกับเครื่องมือใดๆ ให้ตอบว่า "ไม่สามารถตอบคำถามนี้ได้" หรือ "ยังไม่มีข้อมูลในระบบ" และแนะนำให้ติดต่อฝ่าย IT หรือผู้ดูแลระบบ
        วิเคราะห์คำถามอย่างระมัดระวังก่อนเลือกเครื่องมือ""",
        "friendly": """คุณชื่อ 'i-Bot' เด็กชาย อายุ 18 ปี แผนก ICT ของบริษัท NCR Rubber
            ทุกประโยคควรจะมีคำว่า ครับ,นะครับ
        ****ควรใช้เครื่องมือ RAG_Knowledge_Search ก่อนเสมอ*****
         ⚠️ กฎสำคัญ: ส่งคำถามให้เครื่องมือเป็นภาษาไทยเสมอ อย่าแปลเป็นภาษาอังกฤษ

        วิธีใช้เครื่องมือ RAG_Knowledge_Search:
        - รับคำถาม: "ทำไมส่ง email แล้วถึงไม่ไป"
        - ส่งให้เครื่องมือ: "ทำไมส่ง email แล้วถึงไม่ไป" (ภาษาไทยเดิม)
        - ห้ามส่ง: "email not received" หรือ "why email not working"
_
        จะตอบคำถามด้วยความสุภาพและเป็นมิตรเสมอ""",
        
        "professional": "คุณคือ 'i-Bot' ผู้เชี่ยวชาญ IT ของบริษัท NCR Rubber ตอบด้วยความเป็นมืออาชีพและละเอียด",
        "casual": "คุณคือ 'i-Bot' เพื่อนร่วมงานสบายๆ ของบริษัท NCR Rubber พูดแบบเพื่อนๆ ไม่เป็นทางการ"
    }
        prefix = personalities.get(personality, personalities["default"])
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=3,
            # early_stopping_method="force",
            early_stopping_method="generate",
            return_intermediate_steps=True,
            agent_kwargs={
            "prefix": prefix,  # แค่เปลี่ยน prefix เท่านั้น
        }
        #     agent_kwargs={                      # ส่ง argument เพิ่มเติมให้ agent
        #     "prefix": "คุณคือ AI ผู้ช่วยของบริษัท NCR Rubber",
        #     "format_instructions": "ตอบให้กระชับและเป็นกันเอง"
        # },
        )
        return agent
    try:
        agent = get_agent_for_user(user_id)
        response = agent.invoke(question)
        if "intermediate_steps" in response and response["intermediate_steps"]:
            tool_result = response["intermediate_steps"][-1][1]  # [action, observation] -> เอา observation
            return response['output'],tool_result
        else:
            return response.get("output", ""),''
    except Exception as e:
        return f"❌ ไม่สามารถตอบคำถามได้: {str(e)}",''

def phone_search(search_value):
    try:
        print("🔍 กำลังค้นหาหมายเลขโทรศัพท์...")
        df = pd.read_excel('excel/tel_phone.xlsx')
        
        keyword = search_value.strip()
        keyword_lower = keyword.lower()
        
        condition = (
            df['name'].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False) |
            df['suname'].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False) |
            df['dep'].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False) |
            df['dep1'].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False) |
            df['AKA'].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False)
        )

        result = df[condition]

        if not result.empty:
            # ✅ สร้าง JSON structure ที่ LLM เข้าใจง่าย
            contacts = []
            for _, row in result.iterrows():
                contact = {
                    "phone": str(row['tel']) if pd.notna(row['tel']) else None,
                    "name": str(row['name']) if pd.notna(row['name']) else None,
                    "surname": str(row['suname']) if pd.notna(row['suname']) else None,
                    "department": str(row['dep']) if pd.notna(row['dep']) else None,
                    "sub_department": str(row['dep1']) if pd.notna(row['dep1']) else None,
                    "nickname": str(row['AKA']) if pd.notna(row['AKA']) else None,
                    "full_name": f"{str(row['name']) if pd.notna(row['name']) else ''} {str(row['suname']) if pd.notna(row['suname']) else ''}".strip()
                }
                contacts.append(contact)
            
            response_data = {
                "status": "success",
                "search_keyword": keyword,
                "total_found": len(contacts),
                "contacts": contacts,
                "message": f"พบ {len(contacts)} รายการสำหรับ '{keyword}'"
            }
            
            return json.dumps(response_data, ensure_ascii=False, indent=2)
            
        else:
            response_data = {
                "status": "not_found",
                "search_keyword": keyword,
                "total_found": 0,
                "contacts": [],
                "message": f"ไม่พบข้อมูลสำหรับ '{keyword}'",
                "suggestions": [
                    "ลองใช้ชื่อย่อ เช่น: สิป, วราภร, พล",
                    "ลองใช้ชื่อแผนก เช่น: ICT, HR, บัญชี",
                    "ตรวจสอบการพิมพ์ผิด",
                    "ลองใช้ภาษาอังกฤษแบบตัวใหญ่"
                ]
            }
            
            return json.dumps(response_data, ensure_ascii=False, indent=2)
            
    except FileNotFoundError:
        error_data = {
            "status": "error",
            "error_type": "file_not_found",
            "message": "ไม่พบไฟล์ excel/tel_phone.xlsx",
            "suggestion": "กรุณาตรวจสอบไฟล์"
        }
        return json.dumps(error_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_data = {
            "status": "error",
            "error_type": "general_error",
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "suggestion": "กรุณาลองใหม่อีกครั้ง"
        }
        return json.dumps(error_data, ensure_ascii=False, indent=2)
# agent_info = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     memory=memory,
#     handle_parsing_errors=True,
#     return_intermediate_steps=True,
#     verbose=True,
# )

if __name__ == "__main__":
    
    # Define basic tools for testing
    # def dummy_release_memory(text):
    #     return "Memory release function not available in main"
    
    # tools = [
    #     Tool(
    #         name="RAG Database",
    #         func=i_bot_answer,
    #         description="ใช้สำหรับตอบคำถามทางด้านเทคนิค IT ,P2A, Printer, Email,วิธีการถนอมโน๊ตบุ๊ค,วิธีแก้ปัญหาต่างๆ เช่น คำถามเกี่ยวกับโน๊ตบุ๊ค,เครื่องพิมพ์,อีเมล,การใช้งานโปรแกรมต่างๆ"
    #     ),
    #     Tool(
    #         name='ค้นหาข้อมูลในฐานข้อมูล',
    #         func=search_sql_database,
    #         description="ใช้สำหรับค้นหาข้อมูลในฐานข้อมูล SQL Server เช่น ค้นหาหมายเลขคอมพิวเตอร์ ชื่อผู้ใช้ หรือข้อมูลอื่นๆ"
    #     ),
    #     # Tool(
    #     #     name='รัน SQL Query',
    #     #     func=execute_sql_query,
    #     #     description="ใช้สำหรับรัน SQL SELECT query เพื่อดึงข้อมูลเฉพาะจากฐานข้อมูล"
    #     # ),
    #     # Tool(
    #     #     name='ดูรายการตารางฐานข้อมูล',
    #     #     func=get_database_tables,
    #     #     description="ใช้สำหรับดูรายการตารางทั้งหมดในฐานข้อมูล Cedar6_ICT พร้อมจำนวนแถวข้อมูล"
    #     # ),
    #     Tool(
    #         name='ล้างหน่วยความจำแชท',
    #         func=dummy_release_memory,
    #         description="ใช้สำหรับล้างหน่วยความจำแชท"
    #     )
    # ]

    # def get_agent_for_user(user_id):
    #     memory = user_memories[user_id]
    #     agent = initialize_agent(
    #         tools=tools,
    #         llm=llm,
    #         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #         # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #         memory=memory,
    #         handle_parsing_errors=True,
    #         verbose=True,
    #         max_iterations=3,
    #         # early_stopping_method="force",
    #         early_stopping_method="generate",
    #         return_intermediate_steps=True,
    #     #     agent_kwargs={                      # ส่ง argument เพิ่มเติมให้ agent
    #     #     "prefix": "คุณคือ AI ผู้ช่วยของบริษัท NCR Rubber",
    #     #     "format_instructions": "ตอบให้กระชับและเป็นกันเอง"
    #     # },
    #     )
    #     return agent

    text =""


    while text != 'y':
        # user_id = input("User ID: ")
        # if user_id == 'y':
        #     break
        text = input("User message: ")
        try:
            agent = get_agent_ans('test',text)
            response = agent

            print("คำตอบ:", response)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            print('ไม่สามารถตอบคำถามได้ กรุณาลองใหม่อีกครั้ง')
            continue
    # testaaa = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"device": "cuda"})
    # print('done')
    # print('asdfasdfasdasdf')
    # print(i_bot_answer("ไม่สามารถเปิด email ได้"))

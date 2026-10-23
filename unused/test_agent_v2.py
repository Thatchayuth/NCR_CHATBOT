"""
AGENT + RAG Demo System
ระบบตัวอย่างการใช้งาน LangChain Agent ร่วมกับ RAG (Retrieval-Augmented Generation)
สำหรับตอบคำถามด้าน IT/ICT ของบริษัท NCR Rubber
"""

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import requests
import json
from typing import Optional, List

# กำหนดค่าเริ่มต้น
PERSIST_DIRECTORY = "last_persist_db_qwen"
#OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct:latest"  # ใช้ model ที่เหมาะสมกับ embedding
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Global variables สำหรับ lazy loading
_embedding = None
_vectordb = None
_llm = None

class CustomOllamaLLM(LLM):
    """Custom Ollama LLM class สำหรับใช้งานกับ LangChain Agent"""
    model: str
    base_url: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()["response"].strip()
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

def get_embedding():
    """Lazy loading สำหรับ embedding model with fallback"""
    global _embedding
    if _embedding is None:
        print("🔄 โหลด embedding model...")
        try:
            # Try GPU first
            _embedding = HuggingFaceEmbeddings(
                model_name="Qwen/Qwen3-Embedding-0.6B", 
                model_kwargs={"device": "cuda"}
            )
            print("✅ embedding model พร้อมใช้งาน (GPU)")
        except Exception as e:
            print(f"⚠️ ไม่สามารถใช้ GPU ได้: {e}")
            print("🔄 กำลังเปลี่ยนเป็น CPU...")
            try:
                _embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
                print("✅ embedding model พร้อมใช้งาน (CPU)")
            except Exception as e2:
                print(f"❌ ไม่สามารถโหลด embedding model ได้: {e2}")
                raise Exception("Cannot load embedding model")
    return _embedding

def get_vectordb():
    """Lazy loading สำหรับ vector database with error handling"""
    global _vectordb
    if _vectordb is None:
        print("🔄 โหลด vector database...")
        try:
            _vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=get_embedding()
            )
            # Test the database
            count = _vectordb._collection.count()
            print(f"✅ vector database พร้อมใช้งาน ({count} documents)")
        except Exception as e:
            print(f"❌ ไม่สามารถโหลด vector database ได้: {e}")
            print(f"💡 กรุณาตรวจสอบ directory: {PERSIST_DIRECTORY}")
            raise Exception("Cannot load vector database")
    return _vectordb

def get_llm():
    """Lazy loading สำหรับ LLM with connection test"""
    global _llm
    if _llm is None:
        print("🔄 โหลด LLM...")
        try:
            _llm = CustomOllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2,
                top_p=0.50,
                top_k=10, 
                num_predict= 256,
                num_ctx = 3000
            )
            # Test connection
            test_response = _llm("สวัสดี")
            if "Error" not in test_response:
                print("✅ LLM พร้อมใช้งาน")
            else:
                raise Exception(f"LLM connection test failed: {test_response}")
        except Exception as e:
            print(f"❌ ไม่สามารถเชื่อมต่อ LLM ได้: {e}")
            print(f"💡 กรุณาตรวจสอบ:")
            print(f"   - Ollama server ทำงานอยู่หรือไม่ ({OLLAMA_BASE_URL})")
            print(f"   - Model {OLLAMA_MODEL} มีอยู่ในระบบหรือไม่")
            raise Exception("Cannot connect to LLM")
    return _llm

def rag_search_tool(question: str) -> str:
    """
    RAG Tool: ค้นหาข้อมูลจาก Vector Database
    Returns: JSON string ที่มีผลลัพธ์การค้นหา
    """
    try:
        print(f"🔍 กำลังค้นหา: {question}")
        vectordb = get_vectordb()
        
        # ค้นหาเอกสารที่เกี่ยวข้อง
        results = vectordb.similarity_search_with_score(question, k=3)
        
        if not results:
            return json.dumps({
                "status": "not_found",
                "message": "ไม่พบข้อมูลที่เกี่ยวข้อง",
                "question": question
            }, ensure_ascii=False, indent=2)
        
        # จัดรูปแบบผลลัพธ์
        search_results = []
        for doc, score in results:
            metadata = doc.metadata
            result_item = {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "topic": metadata.get("topic", "ไม่ระบุหัวข้อ"),
                "pdf_link": metadata.get("urlfilenew", ""),
                "video_link": metadata.get("clip", ""),
                "confidence_score": round(float(score), 3)
            }
            search_results.append(result_item)
        
        response = {
            "status": "success",
            "question": question,
            "total_results": len(search_results),
            "results": search_results
        }
        
        print(f"✅ พบ {len(search_results)} ผลลัพธ์")
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_response = {
            "status": "error",
            "question": question,
            "error": f"เกิดข้อผิดพลาด: {str(e)}"
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)

def format_rag_results_for_llm(rag_json_result: str) -> str:
    """
    จัดรูปแบบผลลัพธ์จาก RAG Tool ให้เหมาะสำหรับ LLM
    """
    try:
        data = json.loads(rag_json_result)
        
        if data["status"] == "success":
            formatted_text = f"📊 พบข้อมูลที่เกี่ยวข้อง {data['total_results']} รายการ:\n\n"
            
            for i, result in enumerate(data["results"], 1):
                formatted_text += f"🔹 ผลลัพธ์ที่ {i}:\n"
                formatted_text += f"   หัวข้อ: {result['topic']}\n"
                formatted_text += f"   เนื้อหา: {result['content']}\n"
                
                if result["pdf_link"]:
                    formatted_text += f"   📄 ลิงก์ PDF: {result['pdf_link']}\n"
                if result["video_link"]:
                    formatted_text += f"   🎥 ลิงก์วิดีโอ: {result['video_link']}\n"
                
                formatted_text += f"   ความมั่นใจ: {result['confidence_score']}\n\n"
            
            return formatted_text
            
        elif data["status"] == "not_found":
            return f"❌ {data['message']}"
        else:
            return f"❌ เกิดข้อผิดพลาด: {data.get('error', 'ไม่ทราบสาเหตุ')}"
            
    except Exception as e:
        return f"❌ ไม่สามารถประมวลผลข้อมูลได้: {str(e)}"

def enhanced_rag_tool(question: str) -> str:
    """
    Enhanced RAG Tool: ค้นหาข้อมูลและใช้ LLM สร้างคำตอบที่สมบูรณ์
    """
    try:
        # 1. ค้นหาข้อมูลจาก Vector DB
        rag_results = rag_search_tool(question)
        rag_data = json.loads(rag_results)
        
        if rag_data["status"] != "success":
            return f"ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม: {question}"
        
        # 2. จัดรูปแบบข้อมูลสำหรับ LLM
        formatted_context = format_rag_results_for_llm(rag_results)
        
        # 3. สร้าง Prompt สำหรับ LLM
        prompt_template = """
คุณคือ i-Bot ผู้ช่วยด้าน IT ของบริษัท NCR Rubber
โปรดตอบคำถามต่อไปนี้โดยใช้ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล:

คำถาม: {question}

ข้อมูลที่เกี่ยวข้อง:
{context}

คำแนะนำการตอบ:
- ตอบให้กระชับและเข้าใจง่าย
- ใช้ภาษาไทยที่เป็นกันเอง
- ถ้ามีลิงก์ที่เกี่ยวข้อง ให้แนะนำด้วย
- ใส่คำว่า "ครับ" หรือ "นะครับ" ในประโยค
        """
        
        prompt = prompt_template.format(
            question=question,
            context=formatted_context
        )
        
        # 4. เรียก LLM เพื่อสร้างคำตอบ
        llm = get_llm()
        response = llm(prompt)
        
        return response
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"

def simple_calculator(expression: str) -> str:
    """Simple calculator tool สำหรับทดสอบ Multi-tool"""
    try:
        # ตรวจสอบความปลอดภัยก่อน eval
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "❌ อนุญาตเฉพาะตัวเลขและเครื่องหมายทางคณิตศาสตร์เท่านั้น"
        
        result = eval(expression)
        return f"📊 ผลลัพธ์: {expression} = {result}"
    except Exception as e:
        return f"❌ ไม่สามารถคำนวณได้: {str(e)}"

def clear_memory_tool(memory_obj) -> str:
    """Tool สำหรับล้าง memory ของ agent"""
    try:
        if hasattr(memory_obj, 'chat_memory'):
            memory_obj.chat_memory.messages.clear()
            return "🧹 ล้างหน่วยความจำแชทเรียบร้อยแล้ว"
        else:
            return "❌ ไม่สามารถล้างหน่วยความจำได้"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

def get_system_status() -> str:
    """Tool สำหรับตรวจสอบสถานะระบบ"""
    try:
        status = {
            "embedding": "✅ พร้อม" if _embedding else "❌ ยังไม่โหลด",
            "vectordb": "✅ พร้อม" if _vectordb else "❌ ยังไม่โหลด", 
            "llm": "✅ พร้อม" if _llm else "❌ ยังไม่โหลด"
        }
        
        status_text = "🔍 สถานะระบบ:\n"
        for component, state in status.items():
            status_text += f"   {component}: {state}\n"
            
        return status_text
    except Exception as e:
        return f"❌ ไม่สามารถตรวจสอบสถานะได้: {str(e)}"

def create_agent_with_rag():
    """
    สร้าง LangChain Agent ที่มี RAG Tool
    """
    # สร้าง Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    # กำหนด Tools
    tools = [
        Tool(
            name="RAG_Knowledge_Search",
            func=enhanced_rag_tool,
            description="""ใช้สำหรับค้นหาและตอบคำถามด้าน IT/ICT จากฐานความรู้ของบริษัท NCR Rubber 
            เช่น คำถามเกี่ยวกับ laptop, notebook, เครื่องพิมพ์, email, Excel, Zoom, วิธีแก้ปัญหาต่างๆ
            ส่งคำถามเป็นภาษาไทยเสมอ"""
        ),
        Tool(
            name="Basic_RAG_Search", 
            func=rag_search_tool,
            description="ค้นหาข้อมูลพื้นฐานจาก Vector Database (แบบ JSON)"
        ),
        Tool(
            name="Calculator",
            func=simple_calculator,
            description="เครื่องคิดเลขพื้นฐาน สำหรับการคำนวณง่ายๆ เช่น 2+2, 10*5"
        ),
        Tool(
            name="Clear_Memory",
            func=lambda x: clear_memory_tool(memory),
            description="ล้างหน่วยความจำแชท เริ่มการสนทนาใหม่ ใช้เมื่อต้องการลืมบริบทเก่า"
        ),
        Tool(
            name="System_Status",
            func=get_system_status,
            description="ตรวจสอบสถานะการทำงานของระบบ embedding, vectordb, และ llm"
        )
    ]
    
    # สร้าง Agent
    agent = initialize_agent(
        tools=tools,
        llm=get_llm(),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        return_intermediate_steps=True,
        agent_kwargs={
            "prefix": """คุณคือ 'i-Bot' ผู้ช่วยด้าน IT อายุ 18 ปี แผนก ICT ของบริษัท NCR Rubber

🎯 หลักการทำงาน:
1. สำหรับคำถามด้าน IT/เทคนิค → ใช้ RAG_Knowledge_Search เสมอ
2. สำหรับการคำนวณ → ใช้ Calculator
3. ส่งคำถามให้ Tool เป็นภาษาไทยเสมอ
4. ตอบด้วยภาษาไทยที่เป็นกันเอง พร้อมคำว่า "ครับ" หรือ "นะครับ"
5. ถ้าหา tool ที่เหมาะสมไม่ได้ ให้ตอบว่า "ไม่สามารถช่วยในเรื่องนี้ได้ครับ"

⚠️ สำคัญ: วิเคราะห์คำถามให้ดีก่อนเลือก Tool"""
        }
    )
    
    return agent

def test_agent_functionality():
    """ทดสอบการทำงานของ Agent + RAG"""
    print("=" * 60)
    print("🚀 เริ่มทดสอบ AGENT + RAG System")
    print("=" * 60)
    
    # สร้าง Agent
    agent = create_agent_with_rag()
    
    # ตัวอย่างคำถามสำหรับทดสอบ
    test_questions = [
        "วิธีแก้ปัญหา email ส่งไม่ไป",
        "เครื่องพิมพ์ใช้งานไม่ได้ทำอย่างไร",
        "คำนวณ 25 * 4 + 100 / 5",
        "โน๊ตบุ๊คช้า มีวิธีดูแลอย่างไร",
        "ตรวจสอบสถานะระบบ",
        "ล้างแชท"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 คำถามที่ {i}: {question}")
        print("=" * 40)
        
        try:
            response = agent.invoke({"input": question})
            print(f"🤖 คำตอบ: {response.get('output', 'ไม่มีคำตอบ')}")
            
            # แสดง intermediate steps ถ้ามี
            if response.get("intermediate_steps"):
                print("\n🔧 ขั้นตอนการทำงาน:")
                for step in response["intermediate_steps"]:
                    action, observation = step
                    print(f"   Tool ที่ใช้: {action.tool}")
                    print(f"   Input: {action.tool_input}")
                    # print(f"   Output: {observation[:200]}...")
                    
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        
        print("\n" + "=" * 40)

def interactive_chat():
    """โหมดแชทแบบ Interactive with enhanced features"""
    print("=" * 60)
    print("💬 โหมดแชทกับ i-Bot (AGENT + RAG)")
    print("คำสั่งพิเศษ:")
    print("  'quit', 'exit', 'q' = ออกจากระบบ")
    print("  'clear' = ล้างหน่วยความจำแชท")
    print("  'status' = ตรวจสอบสถานะระบบ")
    print("  'help' = แสดงคำแนะนำ")
    print("=" * 60)
    
    agent = create_agent_with_rag()
    
    while True:
        try:
            user_input = input("\n👤 คุณ: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'ออก']:
                print("👋 ขอบคุณที่ใช้บริการครับ!")
                break
                
            if user_input.lower() in ['clear', 'ล้าง']:
                response = agent.invoke({"input": "ล้างแชท"})
                print(f"\n🤖 i-Bot: {response.get('output', 'ล้างหน่วยความจำแล้วครับ')}")
                continue
                
            if user_input.lower() in ['status', 'สถานะ']:
                response = agent.invoke({"input": "ตรวจสอบสถานะระบบ"})
                print(f"\n🤖 i-Bot: {response.get('output', 'ไม่สามารถตรวจสอบสถานะได้')}")
                continue
                
            if user_input.lower() in ['help', 'ช่วยเหลือ']:
                print("""
🤖 i-Bot สามารถช่วยเหลือในเรื่องต่างๆ ได้:
📧 Email: ปัญหาส่งไม่ไป, รับไม่ได้, การตั้งค่า
💻 Computer: โน๊ตบุ๊คช้า, โปรแกรมไม่ทำงาน, วิธีดูแล
🖨️ Printer: เครื่องพิมพ์ไม่พิมพ์, ติดกระดาษ, ไม่เชื่อมต่อ
📊 Software: Excel, Word, PowerPoint, Zoom
🔧 IT Support: วิธีแก้ปัญหาต่างๆ ด้านเทคนิค
🧮 Calculator: การคำนวณทางคณิตศาสตร์

ลองถามคำถามที่สนใจได้เลยครับ!""")
                continue
                
            if not user_input:
                continue
                
            print("\n🤖 i-Bot กำลังคิด...")
            response = agent.invoke({"input": user_input})
            
            print(f"\n🤖 i-Bot: {response.get('output', 'ขอโทษครับ ไม่สามารถตอบได้')}")
            
        except KeyboardInterrupt:
            print("\n\n👋 ออกจากระบบแล้ว!")
            break
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
            print("กรุณาลองใหม่อีกครั้งครับ")

if __name__ == "__main__":
    print("🎯 AGENT + RAG Demo System - Enhanced Version")
    print("=" * 50)
    
    # เลือกโหมดการทำงาน
    print("เลือกโหมดการทำงาน:")
    print("1. 🧪 ทดสอบอัตโนมัติ (Test Mode)")
    print("2. 💬 แชทแบบ Interactive")
    print("3. 🚀 ทดสอบก่อน แล้วเข้าแชท")
    print("4. ℹ️  แสดงข้อมูลระบบ")
    
    try:
        choice = input("\nกรุณาเลือก (1/2/3/4): ").strip()
        
        if choice == "1":
            print("\n🧪 เริ่มโหมดทดสอบ...")
            test_agent_functionality()
            
        elif choice == "2":
            print("\n💬 เริ่มโหมดแชท...")
            interactive_chat()
            
        elif choice == "3":
            print("\n🚀 เริ่มทดสอบก่อน...")
            test_agent_functionality()
            
            proceed = input("\n✅ ทดสอบเสร็จแล้ว ต้องการเข้าโหมดแชทไหม? (y/n): ").strip().lower()
            if proceed in ['y', 'yes', 'ใช่']:
                print("\n💬 เปลี่ยนไปโหมดแชท...")
                interactive_chat()
            else:
                print("👋 ขอบคุณที่ใช้บริการครับ!")
                
        elif choice == "4":
            print("\n📊 ข้อมูลระบบ:")
            print(f"   📁 Vector DB Path: {PERSIST_DIRECTORY}")
            print(f"   🤖 LLM Model: {OLLAMA_MODEL}")
            print(f"   🔗 Ollama URL: {OLLAMA_BASE_URL}")
            print(f"   🛠️  Available Tools: 5 tools")
            print("   📚 Features: RAG Search, Calculator, Memory Management")
            
        else:
            print("❌ กรุณาเลือก 1, 2, 3, หรือ 4")
            print("🔄 เริ่มโหมดแชทโดยอัตโนมัติ...")
            interactive_chat()
            
    except KeyboardInterrupt:
        print("\n\n👋 ออกจากโปรแกรม!")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดในการเริ่มต้น: {str(e)}")
        print("💡 กรุณาตรวจสอบ:")
        print("   - Ollama server ทำงานอยู่หรือไม่")
        print("   - Vector database มีข้อมูลหรือไม่")
        print("   - Python packages ติดตั้งครบหรือไม่")

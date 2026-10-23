from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os,requests
from typing import Optional, List
from langchain.llms.base import LLM
# 1. โหลด PDF และแบ่งข้อความออกเป็นหน้า
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


# loader = PyPDFLoader("test_1.pdf")
# pages = loader.load()

# # 2. แบ่งเนื้อหาเป็นชิ้นเล็ก ๆ
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(pages)

# # 3. ตั้งค่า embedding
# embedding = OllamaEmbeddings(
#     model="scb10x/llama3.1-typhoon2-8b-instruct",  # ต้อง pull ไว้ก่อน
#     base_url="http://192.168.1.97:11434"
# )

# # 4. สร้างฐานข้อมูลเวกเตอร์จากเอกสาร
# vectordb = Chroma.from_documents(docs, embedding=embedding)

persist_directory = "vectordbpdf"

# 📌 โหลดและสร้าง Vector DB หากยังไม่มี
if not os.path.exists(persist_directory):
    loader = PyPDFLoader("test_1.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(
        model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        base_url="http://127.0.0.1:11434"
    )

    vectordb = Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    folder_path = "pdf_folder"  # โฟลเดอร์ที่เก็บ PDF ไว้
    all_docs = []

# โหลดทุก PDF ในโฟลเดอร์
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(full_path)
            documents = loader.load()
            chunks = splitter.split_documents(documents)
            all_docs.extend(chunks)

    # เพิ่มเข้า Vector DB เดิม
    vectordb.add_documents(all_docs)
    vectordb.persist()

else:
    print("✅ Vector DB มีอยู่แล้ว ไม่ต้องโหลดใหม่")


embedding = OllamaEmbeddings(
        model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        base_url="http://127.0.0.1:11434"
    )

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# 5. ตั้งค่า LLM
llm = Ollama(
    model="scb10x/llama3.1-typhoon2-8b-instruct",  # หรือจะลองแค่ "typhoon2" ก็ได้
    base_url="http://127.0.0.1:11434"
)

# 6. สร้าง QA chain ที่ใช้ vector search
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# 7. ถามคำถาม
query = "CSR ของบริษัทนี้มีไหม"
result = qa_chain.invoke({"query": query})

print("คำตอบ:", result["result"])
print("อ้างอิงจาก:", result["source_documents"][0].metadata)

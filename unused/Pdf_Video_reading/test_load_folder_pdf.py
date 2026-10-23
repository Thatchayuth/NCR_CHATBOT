from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os,requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.docstore.document import Document
import fitz  # PyMuPDF
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

class FriendlyLLM(LLM):
    model: str
    base_url: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = super()._call(prompt, stop)
        # Add a friendly tone to the response
        response = f"Hi there! I'd like to help you with that. {response} Would you like me to elaborate on this?"
        return response

    @property
    def _llm_type(self) -> str:
        return "friendly_ollama"

def load_pdf_with_pymupdf(file_path):
    doc = fitz.open(file_path)
    documents = []

    for page in doc:
        text = page.get_text()
        metadata = {"source": file_path, "page": page.number}
        documents.append(Document(page_content=text, metadata=metadata))
        print(text)
    return documents

persist_directory = "vectordbpdf3"

# Load and create Vector DB if not exists
if not os.path.exists(persist_directory):
    folder_path = "ax4_pdf"  # Folder path to load PDFs from
    all_docs = []

    # Load all PDFs in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            # loader = PyPDFLoader(full_path)
            # documents = loader.load()
            documents = load_pdf_with_pymupdf(full_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)
            all_docs.extend(docs)

    # Create Vector DB and persist
    embedding = OllamaEmbeddings(
        model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        base_url="http://127.0.0.1:11434"
    )
    vectordb = Chroma.from_documents(
        all_docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()

# If Vector DB already exists, load it and add new documents
else:
    embedding = OllamaEmbeddings(
        model="scb10x/llama3.1-typhoon2-8b-instruct:latest",
        base_url="http://127.0.0.1:11434"
    )
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

# Set up LLM
llm = Ollama(
    model="scb10x/llama3.1-typhoon2-8b-instruct",  # or try just "typhoon2"
    base_url="http://127.0.0.1:11434"
)
# llm = FriendlyLLM(
#     model="scb10x/llama3.1-typhoon2-8b-instruct",  # or try just "typhoon2"
#     base_url="http://127.0.0.1:11434"
# )
chatbot_style = (
    "คุณคือ 'ก๊อตจิ' แชทบอทประจำบริษัท มีหน้าที่ให้ข้อมูลเกี่ยวกับแผนก IT อย่างสุภาพและเป็นมิตร "
    "หากไม่มีข้อมูลที่เกี่ยวข้องในเอกสารที่ระบบเ" \
    ""
)
# Create QA chain using vector search
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)
# Ask a question
question = "มีข้อมูลอะไรให้ถามบ้าง"
full_query = f"{chatbot_style}\nUser: {question}\nAssistant:"
result = qa_chain.invoke({"query": full_query})
print("คำตอบ:", result["result"])
print("อ้างอิงจาก:", result["source_documents"][0].metadata)
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('source', 'ไม่ทราบที่มา')}")
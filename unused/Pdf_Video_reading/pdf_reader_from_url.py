#%%
import requests
from io import BytesIO
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
# 🔗 URL ของ PDF
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
# ✅ แสดงเนื้อหา (หน้าแรก)


# %%
if __name__ == "__main__":
    url = "C:/Users/teety/Desktop/NCR_CHATBOT/test_1.pdf"  # แทนที่ด้วย URL ของ PDF ที่ต้องการ
    text = get_text_from_url_pdf(url)
    if text:
        print("📄 เนื้อหา PDF:")
        print(text[:500])  # แสดงแค่ 500 ตัวอักษรแรกเพื่อไม่ให้ยาวเกินไป
    else:
        print("ไม่สามารถดึงเนื้อหาจาก PDF ได้")
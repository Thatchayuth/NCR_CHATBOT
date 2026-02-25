"""
สร้าง Database และ Table สำหรับระบบค้นหาเบอร์โทร
รันครั้งเดียวเพื่อ setup database
"""
import pyodbc
from db_config import get_master_connection, get_connection, DB_NAME

def create_database():
    """สร้าง database PhoneDB ถ้ายังไม่มี"""
    conn = get_master_connection()
    conn.autocommit = True
    cursor = conn.cursor()
    
    # ตรวจสอบว่า database มีอยู่แล้วหรือไม่
    cursor.execute(
        "SELECT name FROM sys.databases WHERE name = ?", DB_NAME
    )
    if cursor.fetchone() is None:
        cursor.execute(f"CREATE DATABASE [{DB_NAME}]")
        print(f"สร้าง database '{DB_NAME}' สำเร็จ")
    else:
        print(f"database '{DB_NAME}' มีอยู่แล้ว")
    
    cursor.close()
    conn.close()

def create_tables():
    """สร้าง table all_phone"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # สร้าง table all_phone (ใช้ NVARCHAR เพื่อรองรับภาษาไทย)
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='all_phone' AND xtype='U')
        CREATE TABLE all_phone (
            Id INT IDENTITY(1,1) PRIMARY KEY,
            Site NVARCHAR(50),
            Phone NVARCHAR(100),
            UrgentPhone NVARCHAR(100),
            Department NVARCHAR(200),
            DepartmentEN NVARCHAR(200),
            Section NVARCHAR(200),
            FirstName NVARCHAR(200),
            LastName NVARCHAR(200),
            NickName NVARCHAR(100),
            ShortName NVARCHAR(50),
            Email NVARCHAR(200),
            Position NVARCHAR(300)
        )
    """)
    conn.commit()
    print("สร้าง table 'all_phone' สำเร็จ (หรือมีอยู่แล้ว)")
    
    # สร้าง table com_list (เก็บข้อมูลเครื่องคอมพิวเตอร์)
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='com_list' AND xtype='U')
        CREATE TABLE com_list (
            Id INT IDENTITY(1,1) PRIMARY KEY,
            Client NVARCHAR(100),
            [User] NVARCHAR(200),
            Po_Number NVARCHAR(100),
            PO_Date DATE
        )
    """)
    conn.commit()
    print("สร้าง table 'com_list' สำเร็จ (หรือมีอยู่แล้ว)")
    
    cursor.close()
    conn.close()


def import_com_list_from_excel(excel_file: str = None):
    """Import ข้อมูลเครื่องคอมพิวเตอร์จาก Excel เข้า table com_list"""
    import pandas as pd
    import os
    
    if excel_file is None:
        # ค้นหาไฟล์ com_list.xlsx
        possible_paths = [
            "excel/com_list.xlsx",
            "../excel/com_list.xlsx",
            "E:/I-BOTV2/NCR_CHATBOT/excel/com_list.xlsx",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                excel_file = path
                break
    
    if excel_file is None or not os.path.exists(excel_file):
        print("❌ ไม่พบไฟล์ com_list.xlsx")
        return
    
    print(f"📂 อ่านข้อมูลจาก: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name='Sheet1')
    
    # Standardize column names
    col_map = {'Po Number': 'Po_Number', 'PO Number': 'Po_Number', 'PO_Number': 'Po_Number'}
    df = df.rename(columns=col_map)
    
    # แปลง PO_Date
    if 'PO_Date' in df.columns:
        df['PO_Date'] = pd.to_datetime(df['PO_Date'], errors='coerce')
    
    print(f"✅ อ่านได้ {len(df)} รายการ")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # ล้างข้อมูลเก่าก่อน insert ใหม่
    cursor.execute("DELETE FROM com_list")
    conn.commit()
    print("🗑️ ล้างข้อมูลเก่าเรียบร้อย")
    
    # Insert ข้อมูลใหม่
    insert_sql = """
        INSERT INTO com_list (Client, [User], Po_Number, PO_Date)
        VALUES (?, ?, ?, ?)
    """
    
    success = 0
    errors = 0
    for _, row in df.iterrows():
        try:
            po_date = row.get('PO_Date')
            if pd.isna(po_date):
                po_date = None
            
            cursor.execute(insert_sql, [
                str(row.get('Client', '')).strip() if pd.notna(row.get('Client')) else None,
                str(row.get('User', '')).strip() if pd.notna(row.get('User')) else None,
                str(row.get('Po_Number', '')).strip() if pd.notna(row.get('Po_Number')) else None,
                po_date
            ])
            success += 1
        except Exception as e:
            errors += 1
            print(f"  ⚠️ Error row: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Import สำเร็จ: {success} รายการ, ผิดพลาด: {errors} รายการ")

if __name__ == "__main__":
    print("=" * 50)
    print("  Database Setup - ระบบค้นหาเบอร์โทร & เครื่องคอม")
    print("=" * 50)
    
    print("\n[1/3] สร้าง Database...")
    create_database()
    
    print("\n[2/3] สร้าง Tables...")
    create_tables()
    
    print("\n[3/3] Import ข้อมูลเครื่องคอมพิวเตอร์จาก Excel...")
    import_com_list_from_excel()
    
    print("\n✅ Setup เสร็จสมบูรณ์!")

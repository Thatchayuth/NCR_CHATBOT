import pyodbc
from db_config import get_connection

# Mapping คอลัมน์ SQL -> ชื่อแสดงผล (ไทย)
DISPLAY_MAP = {
    'Site': 'Site',
    'Phone': 'เบอร์โทร',
    'UrgentPhone': 'เบอร์ด่วน',
    'Department': 'ฝ่าย',
    'DepartmentEN': 'Department',
    'Section': 'Section',
    'FirstName': 'ชื่อ',
    'LastName': 'นามสกุล',
    'NickName': 'ชื่อเล่น',
    'ShortName': 'ชื่อย่อ',
    'Email': 'Email',
    'Position': 'ตำแหน่ง',
}

def _rows_to_records(cursor):
    """แปลง cursor rows เป็น list of dict พร้อมชื่อคอลัมน์ไทย"""
    columns = [desc[0] for desc in cursor.description]
    records = []
    for row in cursor.fetchall():
        record = {}
        for col, val in zip(columns, row):
            display_name = DISPLAY_MAP.get(col, col)
            record[display_name] = val
        records.append(record)
    return records

def search_phone(keyword):
    """
    Search for a keyword across all fields in the database (partial match)
    Returns a list of matching records (dictionaries)
    ค้นหาจากทุกคอลัมน์ ยกเว้น Email และ LastName (เหมือนเดิม)
    """
    print(keyword)
    print('#' * 80)
    
    keyword_str = str(keyword).strip()
    like_pattern = f"%{keyword_str}%"
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # ค้นหาแบบ partial match ทุกคอลัมน์ ยกเว้น Email, LastName
    sql = """
        SELECT Site, Phone, UrgentPhone, Department, DepartmentEN, Section,
               FirstName, LastName, NickName, ShortName, Email, Position
        FROM all_phone
        WHERE Site LIKE ?
           OR Phone LIKE ?
           OR UrgentPhone LIKE ?
           OR Department LIKE ?
           OR DepartmentEN LIKE ?
           OR Section LIKE ?
           OR FirstName LIKE ?
           OR NickName LIKE ?
           OR ShortName LIKE ?
           OR Position LIKE ?
    """
    params = [like_pattern] * 10
    cursor.execute(sql, params)
    results = _rows_to_records(cursor)
    
    cursor.close()
    conn.close()
    
    if len(results) > 0:
        print(f"Found {len(results)} matching rows for keyword '{keyword_str}':")
        print("=" * 80)
        for i, record in enumerate(results):
            print(f"Record {i + 1}:")
            for key, value in record.items():
                if value and str(value).strip():
                    print(f"  {key}: {value}")
            print("-" * 40)
        return results
    else:
        print(f"No matches found for keyword '{keyword_str}'")
        return []

def search_phone_specific(phone):
    """
    Search specifically in phone number fields with exact matching
    Returns a list of matching records (dictionaries)
    """
    conn = get_connection()
    cursor = conn.cursor()
    phone_str = str(phone).strip()
    
    sql = """
        SELECT Site, Phone, UrgentPhone, Department, DepartmentEN, Section,
               FirstName, LastName, NickName, ShortName, Email, Position
        FROM all_phone
        WHERE Phone = ? OR UrgentPhone = ?
    """
    cursor.execute(sql, [phone_str, phone_str])
    results = _rows_to_records(cursor)
    
    cursor.close()
    conn.close()
    return results

def search_phone_exact(phone):
    """
    Search for exact phone/email match
    Returns a list of matching records (dictionaries)
    """
    conn = get_connection()
    cursor = conn.cursor()
    phone_str = str(phone).strip()
    
    sql = """
        SELECT Site, Phone, UrgentPhone, Department, DepartmentEN, Section,
               FirstName, LastName, NickName, ShortName, Email, Position
        FROM all_phone
        WHERE Phone = ? OR UrgentPhone = ? OR Email = ?
    """
    cursor.execute(sql, [phone_str, phone_str, phone_str])
    results = _rows_to_records(cursor)
    
    cursor.close()
    conn.close()
    return results

# Interactive mode
if __name__ == "__main__":
    print("=== ระบบค้นหาเบอร์โทร ===")
    print("พิมพ์คำค้นหา (ชื่อ, แผนก, เบอร์โทร ฯลฯ)")
    print("พิมพ์ 'exit' หรือ 'q' เพื่อออก")
    print("="*40)
    
    while True:
        keyword = input("\n ค้นหา: ").strip()
        if keyword.lower() in ('exit', 'q', 'quit'):
            print("ออกจากโปรแกรม")
            break
        if not keyword:
            print("กรุณาพิมพ์คำค้นหา")
            continue
        search_phone(keyword)
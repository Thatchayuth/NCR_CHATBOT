import re
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

# NCR_Phonebook เก็บข้อมูลแบบ normalize หลายตาราง (ไม่มี all_phone แล้ว)
# query ทั้งหมดจึง join ตารางเหล่านี้แล้ว alias คอลัมน์ให้ตรงกับชื่อเดิมใน DISPLAY_MAP
_BASE_SQL = """
    SELECT
        p.Site AS Site,
        n.Ext AS Phone,
        COALESCE(pn.Emergency, n.Emergency) AS UrgentPhone,
        d.Department_TH AS Department,
        d.Department_ENG AS DepartmentEN,
        s.Section_TH AS Section,
        p.EmployName AS FirstName,
        p.EmployLastName AS LastName,
        p.EmployNickName AS NickName,
        p.SpecialName AS ShortName,
        p.Email AS Email,
        p.Position AS Position
    FROM Phone_Person p
    LEFT JOIN Phone_PersonNumber pn ON pn.PersonId = p.UID
    LEFT JOIN Phone_Number n ON n.UID = pn.NumberId
    LEFT JOIN Phone_PersonDept pd ON pd.PersonId = p.UID
    LEFT JOIN MasDepartment d ON d.UID = pd.DepartmentId
    LEFT JOIN MasSection s ON s.UID = pd.SectionId
    WHERE p.Isactive = 1
"""

_SEARCH_FIELDS = [
    "p.Site",
    "n.Ext",
    "COALESCE(pn.Emergency, n.Emergency)",
    "d.Department_TH",
    "d.Department_ENG",
    "s.Section_TH",
    "p.EmployName",
    "p.EmployNickName",
    "p.SpecialName",
    "p.Position",
]

def search_phone(keyword):
    """
    Search for one or more keywords across all fields in the database (partial match)
    รองรับหลายรายชื่อพร้อมกัน โดยคั่นด้วย , หรือ | เช่น "ทองสร้อย, เต๋า"
    Returns a list of matching records (dictionaries)
    ค้นหาจากทุกคอลัมน์ ยกเว้น Email และ LastName (เหมือนเดิม)
    """
    print(keyword)
    print('#' * 80)

    terms = [t.strip() for t in re.split(r'[,|]', str(keyword)) if t.strip()]
    if not terms:
        terms = [str(keyword).strip()]

    conn = get_connection()
    cursor = conn.cursor()

    # ค้นหาแบบ partial match ทุกคอลัมน์ ยกเว้น Email, LastName
    # แต่ละ term เป็นกลุ่ม OR ของทุกคอลัมน์ แล้ว OR รวมกันระหว่าง term
    group_sql = "(" + " OR ".join(f"{field} LIKE ?" for field in _SEARCH_FIELDS) + ")"
    where_clause = " OR ".join([group_sql] * len(terms))
    sql = _BASE_SQL + f" AND ({where_clause})"

    params = []
    for term in terms:
        like_pattern = f"%{term}%"
        params.extend([like_pattern] * len(_SEARCH_FIELDS))

    cursor.execute(sql, params)
    results = _rows_to_records(cursor)

    cursor.close()
    conn.close()

    terms_display = ', '.join(terms)
    if len(results) > 0:
        print(f"Found {len(results)} matching rows for keyword(s) '{terms_display}':")
        print("=" * 80)
        for i, record in enumerate(results):
            print(f"Record {i + 1}:")
            for key, value in record.items():
                if value and str(value).strip():
                    print(f"  {key}: {value}")
            print("-" * 40)
        return results
    else:
        print(f"No matches found for keyword(s) '{terms_display}'")
        return []

def search_phone_specific(phone):
    """
    Search specifically in phone number fields with exact matching
    Returns a list of matching records (dictionaries)
    """
    conn = get_connection()
    cursor = conn.cursor()
    phone_str = str(phone).strip()

    sql = _BASE_SQL + """
        AND (n.Ext = ? OR COALESCE(pn.Emergency, n.Emergency) = ?)
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

    sql = _BASE_SQL + """
        AND (n.Ext = ? OR COALESCE(pn.Emergency, n.Emergency) = ? OR p.Email = ?)
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
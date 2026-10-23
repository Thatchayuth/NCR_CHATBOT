# -*- coding: utf-8 -*-
"""
ค้นหา/สรุปสถิติอุบัติเหตุ จากฐานข้อมูล NCR_iBot_Live ตาราง incidents
- "ขอดูสถิติอุบัติเหตุ"          -> สรุปรวมทั้งหมด
- "สถิติอุบัติเหตุ แผนก LLH"     -> กรองเฉพาะแผนก/สายงาน แล้วสรุป
"""

# ต่อ DB บนเซิร์ฟเวอร์เดิม (NCR606) แต่คนละ database (NCR_iBot_Live)
try:
    from module.db_config import get_connection
except ModuleNotFoundError:
    from db_config import get_connection

INCIDENT_DB = "NCR_iBot_Live"

# คำเรียกแผนกภาษาไทย -> ชื่อแผนกที่ใช้จริงใน DB
# ใช้ตอนที่ผู้ใช้พิมพ์ชื่อไทย เช่น "อุบัติเหตุแผนกสายยาง" (DB เก็บเป็น "LLH")
DEPT_ALIASES = {
    "แผนกสายยาง": "LLH",
    "สายยาง": "LLH",
    "แอลแอลเอช": "LLH",
    "แผนกท่อ": "MBH",
    "ท่อ": "MBH",
    "เอ็มบีเอช": "MBH",
    "แผนกคอมพาวด์": "Compound",
    "คอมพาวด์": "Compound",
    "คอมปาวด์": "Compound",
    "บีเฮดพี": "BHP",
    "บีเอชพี": "BHP",
    "แผนกผลิต": "Production",
    "ผลิต": "Production",
    "แผนกไอซีที": "ICT",
    "ไอซีที": "ICT",
    "แผนกธุรการ": "Admin",
    "ธุรการ": "Admin",
    "แผนกบุคคล": "HR",
    "ทรัพยากรบุคคล": "HR",
    "บุคคล": "HR",
    "แผนกโปรไฟล์": "Profile",
    "โปรไฟล์": "Profile",
    "prof": "Profile",          # คำถามถูก .lower() ก่อนเทียบ จึงครอบทั้ง PROF / Prof
    "แผนกโมล": "Mold",
    "โมลด์": "Mold",
    "โมล": "Mold",
    "ยางขึ้นรูป": "Mold",
}


def _get_conn():
    return get_connection(database=INCIDENT_DB)


def _resolve_alias(q):
    """หาคำเรียกแผนกภาษาไทยในคำถาม คืน (คำไทยล้วนที่เจอ, ชื่อแผนกใน DB)"""
    # เรียงจากคำยาวไปสั้น เพื่อให้ "ทรัพยากรบุคคล" ชนะ "บุคคล"
    for key in sorted(DEPT_ALIASES, key=len, reverse=True):
        if key in q:
            # ตัด "แผนก"/"ฝ่าย" ออก เผื่อ DB เก็บชื่อไทยล้วน เช่น "แผนกผลิต" -> "ผลิต"
            core = key.replace("แผนก", "").replace("ฝ่าย", "").strip() or key
            return core, DEPT_ALIASES[key]
    return None, None


def _detect_scope(query, cur):
    """
    ตรวจว่าผู้ใช้ขอเฉพาะแผนก/สายงานไหนไหม โดยเทียบกับค่าจริงใน DB
    คืน (column, value, op) ถ้าเจอ, หรือ (None, None, None) ถ้าเป็นสถิติรวม
    op เป็น "=" (ตรงตัว) หรือ "LIKE" (ครอบทุกสายย่อย เช่น ผลิต 1, ผลิต 2)
    """
    q = (query or "").lower()

    # ดึงค่าจริงใน DB มาเก็บไว้ก่อน (department ก่อน แล้วค่อย division)
    values = {}
    for col in ("department", "division"):
        cur.execute(f"SELECT DISTINCT {col} FROM incidents WHERE {col} IS NOT NULL")
        values[col] = [r[0] for r in cur.fetchall() if r[0] and str(r[0]).strip()]

    # 1) เทียบตรงกับชื่อที่มีใน DB (เรียงจากชื่อยาวไปสั้น เพื่อจับตัวที่เฉพาะเจาะจงกว่าก่อน)
    for col in ("department", "division"):
        for val in sorted(values[col], key=lambda v: len(v), reverse=True):
            if val.lower() in q:
                return col, val, "="

    # 2) ผู้ใช้พิมพ์ชื่อแผนกเป็นภาษาไทย -> แปลงเป็นชื่อที่ DB ใช้
    thai_word, dept = _resolve_alias(q)
    if dept:
        # 2.1) หาแผนกใน DB ที่ตรงกับชื่อที่แปลงได้ เช่น "สายยาง" -> "LLH"
        for col in ("department", "division"):
            for val in sorted(values[col], key=lambda v: len(v), reverse=True):
                if dept.lower() in val.lower():
                    return col, val, "="
        # 2.2) DB อาจเก็บเป็นภาษาไทย เช่น "ผลิต" -> "ผลิต 1"
        #      ใช้ LIKE เพื่อรวมทุกสายย่อย (ผลิต 1, ผลิต 2, ...) ในครั้งเดียว
        for col in ("department", "division"):
            if any(thai_word in val.lower() for val in values[col]):
                return col, thai_word, "LIKE"
        # 2.3) รู้ว่าถามถึงแผนกไหน แต่ยังไม่มีข้อมูลใน DB
        #      คืนชื่อแผนกไป เพื่อให้ตอบว่า "ไม่พบข้อมูลของแผนกนี้" แทนที่จะโชว์สถิติรวม
        return "department", dept, "="

    return None, None, None


def _fetch_stats(cur, where_sql="", params=None):
    """ดึงตัวเลขสรุปพื้นฐานภายใต้เงื่อนไข where ที่กำหนด"""
    params = params or []

    cur.execute(f"""
        SELECT
            COUNT(*)                              AS total,
            ISNULL(SUM(has_injured), 0)           AS injured,
            ISNULL(SUM(is_major), 0)              AS major
        FROM incidents
        {where_sql}
    """, params)
    total, injured, major = cur.fetchone()
    total = total or 0
    injured = injured or 0
    major = major or 0
    minor = total - major

    return {"total": total, "injured": injured, "major": major, "minor": minor}


def _fetch_group(cur, group_col, where_sql="", params=None, limit=10):
    """สรุปจำนวนแยกตามคอลัมน์ที่ระบุ (เช่น department, accident_type)"""
    params = params or []
    cur.execute(f"""
        SELECT TOP {int(limit)} {group_col} AS k, COUNT(*) AS c
        FROM incidents
        {where_sql}
        GROUP BY {group_col}
        ORDER BY c DESC
    """, params)
    return [(r[0] if r[0] else "-", r[1]) for r in cur.fetchall()]


def _fetch_recent(cur, where_sql="", params=None, limit=5):
    """รายการอุบัติเหตุล่าสุด"""
    params = params or []
    cur.execute(f"""
        SELECT TOP {int(limit)} date, department, accident_type, body_part, classification
        FROM incidents
        {where_sql}
        ORDER BY date DESC, id DESC
    """, params)
    return cur.fetchall()


def search_incident(query):
    """
    ฟังก์ชันหลัก: รับคำถาม -> คืนข้อความสรุปสถิติอุบัติเหตุ (string)
    """
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        col, val, op = _detect_scope(query, cur)

        if col and val:
            if op == "LIKE":
                where_sql = f"WHERE {col} LIKE ?"
                params = [f"%{val}%"]
            else:
                where_sql = f"WHERE {col} = ?"
                params = [val]
            title = f"📊 สถิติอุบัติเหตุ - {val}"
        else:
            where_sql = ""
            params = []
            title = "📊 สถิติอุบัติเหตุ (ทั้งหมด)"

        stats = _fetch_stats(cur, where_sql, params)

        if stats["total"] == 0:
            if col and val:
                return f"ไม่พบข้อมูลอุบัติเหตุของ {val} ครับ"
            return "ยังไม่มีข้อมูลอุบัติเหตุในระบบครับ"

        lines = [title, ""]
        lines.append(f"• รวมทั้งหมด: {stats['total']} ครั้ง")
        lines.append(f"• มีผู้บาดเจ็บ: {stats['injured']} ครั้ง")
        lines.append(f"• ระดับ Major: {stats['major']} ครั้ง | Minor: {stats['minor']} ครั้ง")

        # ถ้าเป็นสถิติรวม -> แยกตามแผนกด้วย
        if not (col and val):
            by_dept = _fetch_group(cur, "department", where_sql, params)
            if by_dept:
                lines.append("")
                lines.append("แยกตามแผนก:")
                for name, cnt in by_dept:
                    lines.append(f"• {name}: {cnt} ครั้ง")

        # แยกตามประเภทอุบัติเหตุ
        by_type = _fetch_group(cur, "accident_type", where_sql, params)
        if by_type:
            lines.append("")
            lines.append("แยกตามประเภท:")
            for name, cnt in by_type:
                lines.append(f"• {name}: {cnt} ครั้ง")

        # รายการล่าสุด
        recent = _fetch_recent(cur, where_sql, params)
        if recent:
            lines.append("")
            lines.append("รายการล่าสุด:")
            for row in recent:
                d, dept, atype, part, cls = row
                part = part if part and str(part).strip() not in ("", "-") else "-"
                lines.append(f"• {d} | {dept} | {atype} | {part} | {cls}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ ดึงข้อมูลสถิติอุบัติเหตุไม่สำเร็จครับ: {e}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    for q in ["ขอดูสถิติอุบัติเหตุ", "สถิติอุบัติเหตุแผนก LLH", "อุบัติเหตุแผนก MBH"]:
        print("=" * 60)
        print("Q:", q)
        print(search_incident(q))

"""
Phone Directory Exact Match Search
Performs exact matching on Excel phone directory data
Supports multiple search criteria with flexible field mapping
"""

import pandas as pd
import json
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path


class PhoneDirectoryExactSearch:
    """
    Phone directory search engine that performs exact matching
    using Excel file as the database source
    """
    
    def __init__(self, excel_file_path: str, json_backup_path: Optional[str] = None):
        """
        Initialize the phone search engine
        
        Args:
            excel_file_path: Path to the Excel file containing phone directory
            json_backup_path: Optional JSON backup file path
        """
        self.excel_file_path = Path(excel_file_path)
        self.json_backup_path = Path(json_backup_path) if json_backup_path else None
        self.data_df = None
        self.data_records = []
        
        # Natural language patterns for Thai
        self.phone_patterns = [
            r'(?:ขอ)?เบอร์(?:โทร)?(?:ของ|ให้)?(.+)',
            r'(?:หา)?เบอร์(?:โทร)?(.+)',

            r'โทร(?:หา)?(.+)',

            r'ติดต่อ(.+)',

            r'เบอร์(.+)'
        ]
        
        self.name_patterns = [
            r'(?:ขอ)?ข้อมูล(?:ของ)?(.+)',

            r'(?:หา)?คน(?:ชื่อ)?(.+)',

            r'(?:หา)?ชื่อ(.+)',

            r'(?:ข้อมูล)?พนักงาน(?:ชื่อ)?(.+)'
        ]
        
        self.dept_patterns = [
            r'(?:ขอ)?(?:คนใน)?ฝ่าย(.+)',

            r'(?:หา)?(?:พนักงาน)?(?:ใน)?แผนก(.+)',

            r'(?:รายชื่อ)?(?:ใน)?department(.+)',

            r'(?:หา)?(?:คนที่)?ทำงาน(?:ใน)?(.+)',

        ]
        
        self.site_patterns = [
            r'(?:ขอ)?(?:รายชื่อ)?(?:ใน)?สาขา(.+)',

            r'(?:หา)?(?:คนที่)?site(.+)',

            r'(?:พนักงาน)?(?:ที่)?สาขา(.+)'
        ]
        
        self.load_data()
    
    def load_data(self) -> bool:
        """
        Load phone directory data from Excel file
        Falls back to JSON if Excel loading fails
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Try loading from Excel first
            if self.excel_file_path.exists():
                self.data_df = pd.read_excel(self.excel_file_path)
                self.data_records = self.data_df.to_dict('records')
                print(f"✅ Loaded {len(self.data_records)} records from Excel: {self.excel_file_path.name}")
                return True
                
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            
        # Fallback to JSON if available
        if self.json_backup_path and self.json_backup_path.exists():
            try:
                with open(self.json_backup_path, 'r', encoding='utf-8') as f:
                    self.data_records = json.load(f)
                    self.data_df = pd.DataFrame(self.data_records)
                print(f"✅ Loaded {len(self.data_records)} records from JSON backup: {self.json_backup_path.name}")
                return True
            except Exception as e:
                print(f"❌ Error loading JSON backup: {e}")
        
        print("❌ Failed to load any data source")
        return False
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching
        
        Args:
            text: Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not isinstance(text, str):
            return str(text).strip()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def search_by_phone(self, phone_number: Union[str, int]) -> List[Dict[str, Any]]:
        """
        Search by phone number (exact match)
        
        Args:
            phone_number: Phone number to search for
            
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        phone_str = str(phone_number).strip()
        results = []
        
        for record in self.data_records:
            # Check main phone number
            if str(record.get('เบอร์โทร', '')).strip() == phone_str:
                results.append(record)
            # Check emergency/direct phone
            elif str(record.get('เบอร์ด่วน', '')).strip() == phone_str:
                results.append(record)
        
        return results
    
    def search_by_name(self, name: str, search_type: str = 'any') -> List[Dict[str, Any]]:
        """
        Search by name (exact match in different name fields)
        
        Args:
            name: Name to search for
            search_type: 'any', 'first', 'last', 'nickname', 'abbreviation'
            
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        name_normalized = self.normalize_text(name)
        results = []
        
        for record in self.data_records:
            match_found = False
            
            if search_type in ['any', 'first']:
                first_name = self.normalize_text(str(record.get('ชื่อ', '')))
                if first_name == name_normalized:
                    match_found = True
            
            if search_type in ['any', 'last']:
                last_name = self.normalize_text(str(record.get('นามสกุล', '')))
                if last_name == name_normalized:
                    match_found = True
            
            if search_type in ['any', 'nickname']:
                nickname = self.normalize_text(str(record.get('ชื่อเล่น', '')))
                if nickname == name_normalized:
                    match_found = True
            
            if search_type in ['any', 'abbreviation']:
                abbrev = self.normalize_text(str(record.get('ชื่อย่อ', '')))
                if abbrev == name_normalized:
                    match_found = True
            
            if match_found:
                results.append(record)
        
        return results
    
    def search_by_department(self, department: str, lang: str = 'any') -> List[Dict[str, Any]]:
        """
        Search by department (exact match)
        
        Args:
            department: Department name to search for
            lang: 'any', 'thai', 'english'
            
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        dept_normalized = self.normalize_text(department)
        results = []
        
        for record in self.data_records:
            match_found = False
            
            if lang in ['any', 'thai']:
                thai_dept = self.normalize_text(str(record.get('ฝ่าย', '')))
                if thai_dept == dept_normalized:
                    match_found = True
            
            if lang in ['any', 'english']:
                eng_dept = self.normalize_text(str(record.get('Department', '')))
                if eng_dept == dept_normalized:
                    match_found = True
            
            if match_found:
                results.append(record)
        
        return results
    
    def search_by_position(self, position: str) -> List[Dict[str, Any]]:
        """
        Search by position/title (exact match)
        
        Args:
            position: Position/title to search for
            
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        pos_normalized = self.normalize_text(position)
        results = []
        
        for record in self.data_records:
            record_pos = self.normalize_text(str(record.get('ตำแหน่ง', '')))
            if record_pos == pos_normalized:
                results.append(record)
        
        return results
    
    def search_by_site(self, site: str) -> List[Dict[str, Any]]:
        """
        Search by site/location (exact match)
        
        Args:
            site: Site code to search for
            
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        site_normalized = self.normalize_text(site)
        results = []
        
        for record in self.data_records:
            record_site = self.normalize_text(str(record.get('Site', '')))
            if record_site == site_normalized:
                results.append(record)
        
        return results
    
    def multi_field_search(self, **criteria) -> List[Dict[str, Any]]:
        """
        Search using multiple criteria (AND logic)
        
        Args:
            **criteria: Field-value pairs for searching
                       Supported: phone, name, department, position, site, section
        
        Returns:
            List of matching records
        """
        if not self.data_records:
            return []
        
        results = self.data_records.copy()
        
        # Apply phone filter
        if 'phone' in criteria:
            phone_results = self.search_by_phone(criteria['phone'])
            results = [r for r in results if r in phone_results]
        
        # Apply name filter
        if 'name' in criteria:
            name_results = self.search_by_name(criteria['name'])
            results = [r for r in results if r in name_results]
        
        # Apply department filter
        if 'department' in criteria:
            dept_results = self.search_by_department(criteria['department'])
            results = [r for r in results if r in dept_results]
        
        # Apply position filter
        if 'position' in criteria:
            pos_results = self.search_by_position(criteria['position'])
            results = [r for r in results if r in pos_results]
        
        # Apply site filter
        if 'site' in criteria:
            site_results = self.search_by_site(criteria['site'])
            results = [r for r in results if r in site_results]
        
        # Apply section filter
        if 'section' in criteria:
            section_normalized = self.normalize_text(criteria['section'])
            results = [r for r in results 
                      if self.normalize_text(str(r.get('Section', ''))) == section_normalized]
        
        return results
    
    def format_result(self, record: Dict[str, Any]) -> str:
        """
        Format a single search result for display
        
        Args:
            record: Record dictionary
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append(f"👤 {record.get('ชื่อ', '')} {record.get('นามสกุล', '')}")
        
        if record.get('ชื่อเล่น'):
            lines.append(f"   ชื่อเล่น: {record.get('ชื่อเล่น')}")
        
        if record.get('ชื่อย่อ'):
            lines.append(f"   ชื่อย่อ: {record.get('ชื่อย่อ')}")
        
        lines.append(f"📞 เบอร์โทร: {record.get('เบอร์โทร', 'N/A')}")
        
        if record.get('เบอร์ด่วน'):
            lines.append(f"📞 เบอร์ด่วน: {record.get('เบอร์ด่วน')}")
        
        lines.append(f"🏢 ฝ่าย: {record.get('ฝ่าย', 'N/A')} ({record.get('Department', 'N/A')})")
        lines.append(f"📍 แผนก: {record.get('Section', 'N/A')}")
        lines.append(f"💼 ตำแหน่ง: {record.get('ตำแหน่ง', 'N/A')}")
        lines.append(f"🏠 สาขา: {record.get('Site', 'N/A')}")
        
        if record.get('ใช้ร่วมกัน'):
            lines.append(f"📋 หมายเหตุ: {record.get('ใช้ร่วมกัน')}")
        
        return '\n'.join(lines)
    
    def search_and_format(self, search_type: str, search_value: str, **kwargs) -> str:
        """
        Unified search and format function
        
        Args:
            search_type: Type of search ('phone', 'name', 'department', 'position', 'site', 'multi')
            search_value: Value to search for
            **kwargs: Additional parameters for specific search types
            
        Returns:
            Formatted search results
        """
        results = []
        
        if search_type == 'phone':
            results = self.search_by_phone(search_value)
        elif search_type == 'name':
            name_type = kwargs.get('name_type', 'any')
            results = self.search_by_name(search_value, name_type)
        elif search_type == 'department':
            lang = kwargs.get('lang', 'any')
            results = self.search_by_department(search_value, lang)
        elif search_type == 'position':
            results = self.search_by_position(search_value)
        elif search_type == 'site':
            results = self.search_by_site(search_value)
        elif search_type == 'multi':
            # search_value should be a dict for multi-field search
            if isinstance(search_value, dict):
                results = self.multi_field_search(**search_value)
            else:
                return "❌ Multi-field search requires a dictionary of criteria"
        
        if not results:
            return f"❌ ไม่พบข้อมูลที่ตรงกับ '{search_value}'"
        
        output_lines = [f"✅ พบ {len(results)} รายการ:"]
        output_lines.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            output_lines.append(f"\n📋 รายการที่ {i}:")
            output_lines.append(self.format_result(result))
            if i < len(results):
                output_lines.append("-" * 30)
        
        return '\n'.join(output_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the phone directory
        
        Returns:
            Dictionary with statistics
        """
        if not self.data_records:
            return {"error": "No data loaded"}
        
        stats = {
            "total_records": len(self.data_records),
            "sites": len(set(r.get('Site', '') for r in self.data_records if r.get('Site'))),
            "departments": len(set(r.get('ฝ่าย', '') for r in self.data_records if r.get('ฝ่าย'))),
            "sections": len(set(r.get('Section', '') for r in self.data_records if r.get('Section'))),
        }
        
        # Count records with phone numbers
        with_phone = sum(1 for r in self.data_records if r.get('เบอร์โทร'))
        with_emergency = sum(1 for r in self.data_records if r.get('เบอร์ด่วน'))
        
        stats.update({
            "with_phone": with_phone,
            "with_emergency_phone": with_emergency,
            "phone_coverage": f"{(with_phone/len(self.data_records)*100):.1f}%"
        })
        
        return stats
    
    def parse_natural_language(self, query: str) -> Tuple[str, str]:
        """
        Parse natural language query to determine search type and value
        
        Args:
            query: Natural language query in Thai
            
        Returns:
            Tuple of (search_type, search_value)
        """
        query = query.strip()
        
        # Check if it's a pure number (phone number)
        if re.match(r'^\d+$', query):
            return 'phone', query
        
        # Check for phone number patterns
        for pattern in self.phone_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                search_value = match.group(1).strip()
                # If the captured value is a number, search by phone
                if re.match(r'^\d+$', search_value):
                    return 'phone', search_value
                # Otherwise, search by name (person whose phone we want)
                else:
                    return 'name', search_value
        
        # Check for department patterns
        for pattern in self.dept_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                search_value = match.group(1).strip()
                return 'department', search_value
        
        # Check for site patterns
        for pattern in self.site_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                search_value = match.group(1).strip()
                return 'site', search_value
        
        # Check for name patterns
        for pattern in self.name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                search_value = match.group(1).strip()
                return 'name', search_value
        
        # Default: treat as name search
        return 'name', query
    
    def smart_search(self, query: str) -> str:
        """
        Perform smart search using natural language parsing
        
        Args:
            query: Natural language query
            
        Returns:
            Formatted search results
        """
        search_type, search_value = self.parse_natural_language(query)
        
        # Show what we detected
        type_names = {
            'phone': 'เบอร์โทร',
            'name': 'ชื่อ',
            'department': 'ฝ่าย/แผนก',
            'site': 'สาขา'
        }
        
        print(f"🔍 ตรวจพบ: ค้นหา{type_names.get(search_type, 'ชื่อ')} '{search_value}'")
        
        return self.search_and_format(search_type, search_value)
    
    def get_statistics(self) -> str:
        """
        Get formatted statistics about the phone directory
        
        Returns:
            Formatted statistics string
        """
        stats = self.get_stats()
        if "error" in stats:
            return "❌ No data available"
        
        lines = []
        lines.append("📊 Phone Directory Statistics:")
        lines.append("=" * 40)
        lines.append(f"📋 Total Records: {stats['total_records']}")
        lines.append(f"🏢 Sites: {stats['sites']}")
        lines.append(f"🏛️ Departments: {stats['departments']}")
        lines.append(f"📍 Sections: {stats['sections']}")
        lines.append(f"📞 With Phone: {stats['with_phone']} ({stats['phone_coverage']})")
        lines.append(f"🚨 With Emergency Phone: {stats['with_emergency_phone']}")
        
        return '\n'.join(lines)
    
    def interactive_cli(self):
        """Interactive CLI for phone directory search with natural language support"""
        print("📞 Phone Directory Search System")
        print("=" * 50)
        print("Commands:")
        print("  phone <number>     - Search by phone number")
        print("  name <name>        - Search by name")
        print("  dept <department>  - Search by department")
        print("  site <site>        - Search by site")
        print("  stats              - Show statistics")
        print("  help               - Show this help")
        print("  exit               - Exit the program")
        print("\n🌟 Natural Language Search (NEW!):")
        print("  ขอเบอร์ชื่อสมชาย      - Find phone for 'สมชาย'")
        print("  เบอร์0812345678      - Find info for phone '0812345678'")
        print("  ขอรายชื่อฝ่าย IT     - Find people in IT department")
        print("  พนักงานสาขากรุงเทพ    - Find people at Bangkok site")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Search: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    print("\nCommands:")
                    print("  phone <number>     - Search by phone number")
                    print("  name <name>        - Search by name")
                    print("  dept <department>  - Search by department")
                    print("  site <site>        - Search by site")
                    print("  stats              - Show statistics")
                    print("\n🌟 Natural Language Examples:")
                    print("  ขอเบอร์ชื่อสมชาย      - Find phone for 'สมชาย'")
                    print("  เบอร์0812345678      - Find info for phone '0812345678'")
                    print("  ขอรายชื่อฝ่าย IT     - Find people in IT department")
                    print("  พนักงานสาขากรุงเทพ    - Find people at Bangkok site")
                    continue
                
                if user_input.lower() == 'stats':
                    print(self.get_statistics())
                    continue
                
                # Check if it's a natural language query or structured command
                parts = user_input.split(maxsplit=1)
                
                # If it looks like a structured command (command + parameter)
                if len(parts) == 2 and parts[0].lower() in ['phone', 'name', 'dept', 'department', 'site']:
                    command = parts[0].lower()
                    search_term = parts[1]
                    
                    if command == 'phone':
                        result = self.search_and_format('phone', search_term)
                    elif command == 'name':
                        result = self.search_and_format('name', search_term)
                    elif command in ['dept', 'department']:
                        result = self.search_and_format('department', search_term)
                    elif command == 'site':
                        result = self.search_and_format('site', search_term)
                else:
                    # Use natural language processing
                    result = self.smart_search(user_input)
                
                print(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """
    Demo/test function for the phone search engine
    """
    # Initialize the search engine
    excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
    json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
    
    search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
    
    if not search_engine.data_records:
        print("❌ No data loaded. Please check file paths.")
        return
    
    # Display statistics
    stats = search_engine.get_stats()
    print("📊 Phone Directory Statistics:")
    print(f"   Total Records: {stats['total_records']}")
    print(f"   Sites: {stats['sites']}")
    print(f"   Departments: {stats['departments']}")
    print(f"   Phone Coverage: {stats['phone_coverage']}")
    print()
    
    # Demo searches
    print("🔍 Demo Searches:")
    print("=" * 50)
    
    # Search by phone number
    print("\n1. Search by phone number (701):")
    print(search_engine.search_and_format('phone', '701'))
    
    # Search by name
    print("\n2. Search by first name (ทองสร้อย):")
    print(search_engine.search_and_format('name', 'ทองสร้อย', name_type='first'))
    
    # Search by department
    print("\n3. Search by department (ฝ่ายธุรการ):")
    print(search_engine.search_and_format('department', 'ฝ่ายธุรการ', lang='thai'))
    
    # Search by site
    print("\n4. Search by site (RY):")
    results = search_engine.search_by_site('RY')
    print(f"✅ Found {len(results)} records for site 'RY'")
    
    # Multi-field search
    print("\n5. Multi-field search (Site: RY, Department: Admin):")
    multi_criteria = {'site': 'RY', 'department': 'Admin'}
    print(search_engine.search_and_format('multi', multi_criteria))


def interactive_search():
    """
    Interactive search mode
    """
    excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
    json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
    
    search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
    
    if not search_engine.data_records:
        print("❌ No data loaded. Please check file paths.")
        return
    
    print("🔍 Interactive Phone Directory Search")
    print("=" * 40)
    print("Commands:")
    print("  phone <number>     - Search by phone number")
    print("  name <name>        - Search by name")
    print("  dept <department>  - Search by department")
    print("  site <site>        - Search by site")
    print("  stats              - Show statistics")
    print("  quit               - Exit")
    print()
    
    while True:
        try:
            command = input("🔍 Enter search command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if command.lower() == 'stats':
                stats = search_engine.get_stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            parts = command.split(maxsplit=1)
            if len(parts) != 2:
                print("❌ Invalid command format. Use: <type> <value>")
                continue
            
            search_type, search_value = parts
            
            if search_type.lower() in ['phone', 'p']:
                result = search_engine.search_and_format('phone', search_value)
            elif search_type.lower() in ['name', 'n']:
                result = search_engine.search_and_format('name', search_value)
            elif search_type.lower() in ['dept', 'department', 'd']:
                result = search_engine.search_and_format('department', search_value)
            elif search_type.lower() in ['site', 's']:
                result = search_engine.search_and_format('site', search_value)
            else:
                print("❌ Unknown search type. Use: phone, name, dept, site")
                continue
            
            print(result)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def search_phone_directory(query: str, excel_path: str = None, json_backup: str = None) -> str:
    """
    Simple function for LLM integration - searches phone directory with natural language
    
    Args:
        query: Natural language search query (e.g., "ขอเบอร์ทองสร้อย", "หาคนในฝ่าย IT")
        excel_path: Path to Excel file (optional, uses default if None)
        json_backup: Path to JSON backup (optional, uses default if None)
    
    Returns:
        str: Formatted search results ready for LLM response
    
    Examples:
        >>> search_phone_directory("ขอเบอร์ทองสร้อย")
        >>> search_phone_directory("หาคนในฝ่าย IT")
        >>> search_phone_directory("701")
        >>> search_phone_directory("สาขา RY")
    """
    try:
        # Use default paths if not provided
        if excel_path is None:
            excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
        if json_backup is None:
            json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
        
        # Initialize search engine
        search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
        
        if not search_engine.data_records:
            return "❌ ไม่สามารถโหลดข้อมูลสมุดโทรศัพท์ได้ กรุณาตรวจสอบไฟล์ข้อมูล"
        
        # Perform smart search
        result = search_engine.smart_search(query)
        
        return result
        
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}"


def search_phone_directory_simple(query: str) -> Dict[str, Any]:
    """
    Simple function that returns structured data for LLM processing
    
    Args:
        query: Natural language search query
    
    Returns:
        Dict containing search results and metadata
    
    Structure:
        {
            "success": bool,
            "query": str,
            "search_type": str,
            "search_value": str,
            "results_count": int,
            "results": List[Dict],
            "formatted_response": str,
            "error": str (if any)
        }
    """
    try:
        # Default paths
        excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
        json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
        
        # Initialize search engine
        search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
        
        if not search_engine.data_records:
            return {
                "success": False,
                "query": query,
                "error": "ไม่สามารถโหลดข้อมูลสมุดโทรศัพท์ได้",
                "results_count": 0,
                "results": [],
                "formatted_response": "❌ ไม่สามารถโหลดข้อมูลสมุดโทรศัพท์ได้"
            }
        
        # Parse the query
        search_type, search_value = search_engine.parse_natural_language(query)
        
        # Perform search based on type
        if search_type == 'phone':
            results = search_engine.search_by_phone(search_value)
        elif search_type == 'name':
            results = search_engine.search_by_name(search_value)
        elif search_type == 'department':
            results = search_engine.search_by_department(search_value)
        elif search_type == 'site':
            results = search_engine.search_by_site(search_value)
        else:
            results = search_engine.search_by_name(search_value)  # Default to name search
        
        # Format results
        formatted_response = search_engine.search_and_format(search_type, search_value)
        
        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "search_value": search_value,
            "results_count": len(results),
            "results": results,
            "formatted_response": formatted_response,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"เกิดข้อผิดพลาด: {str(e)}",
            "results_count": 0,
            "results": [],
            "formatted_response": f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}"
        }


def get_phone_directory_info() -> Dict[str, Any]:
    """
    Get information about the phone directory for LLM context
    
    Returns:
        Dict with directory statistics and available search types
    """
    try:
        excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
        json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
        
        search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
        
        if not search_engine.data_records:
            return {"error": "ไม่สามารถโหลดข้อมูลได้"}
        
        stats = search_engine.get_stats()
        
        # Get sample data for LLM understanding
        sample_record = search_engine.data_records[0] if search_engine.data_records else {}
        
        return {
            "stats": stats,
            "available_fields": list(sample_record.keys()) if sample_record else [],
            "search_capabilities": [
                "ค้นหาด้วยเบอร์โทรศัพท์",
                "ค้นหาด้วยชื่อ-นามสกุล",
                "ค้นหาด้วยชื่อเล่น",
                "ค้นหาด้วยชื่อย่อ",
                "ค้นหาด้วยฝ่าย/แผนก",
                "ค้นหาด้วยสาขา",
                "ค้นหาด้วยตำแหน่งงาน"
            ],
            "example_queries": [
                "ขอเบอร์ทองสร้อย",
                "หาเบอร์ก้อย",
                "เบอร์701",
                "คนในฝ่าย IT",
                "พนักงานสาขา RY",
                "หาผู้จัดการ"
            ]
        }
        
    except Exception as e:
        return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}


def test_llm_functions():
    """Test the LLM-ready functions"""
    print("🧪 Testing LLM Functions")
    print("=" * 40)
    
    # Test 1: Simple search_phone function (NEW!)
    print("\n1. Testing search_phone() - Simple Data Return:")
    data1 = search_phone("ขอเบอร์ทองสร้อย")
    print(f"   Found {len(data1)} records")
    if data1:
        first_record = data1[0]
        print(f"   First result: {first_record.get('ชื่อ', '')} {first_record.get('นามสกุล', '')} - เบอร์: {first_record.get('เบอร์โทร', 'N/A')}")
    
    # Test 2: Search by phone number
    print("\n2. Testing search_phone('701'):")
    data2 = search_phone("701")
    print(f"   Found {len(data2)} records for phone '701'")
    if data2:
        for record in data2:
            print(f"   - {record.get('ชื่อ', '')} {record.get('นามสกุล', '')} ({record.get('ชื่อเล่น', '')})")
    
    # Test 3: Search by department
    print("\n3. Testing search_phone('ฝ่ายธุรการ'):")
    data3 = search_phone("ฝ่ายธุรการ")
    print(f"   Found {len(data3)} records in ฝ่ายธุรการ")
    if data3:
        print(f"   Sample: {data3[0].get('ชื่อ', '')} {data3[0].get('นามสกุล', '')}")
    
    # Test 4: Original functions
    print("\n4. Testing search_phone_directory_simple():")
    result4 = search_phone_directory_simple("เบอร์701")
    print(f"   Success: {result4['success']}")
    print(f"   Results Count: {result4['results_count']}")
    print(f"   Search Type: {result4['search_type']}")
    
    # Test 5: Directory info
    print("\n5. Testing get_phone_directory_info():")
    info = get_phone_directory_info()
    if "stats" in info:
        print(f"   Total Records: {info['stats']['total_records']}")
        print(f"   Available Fields: {len(info['available_fields'])}")
    
    print("\n✅ LLM Function Tests Complete")
    print("\n📋 Usage Examples:")
    print("   data = search_phone('ขอเบอร์ทองสร้อย')  # Search by name")
    print("   data = search_phone('701')              # Search by phone")
    print("   data = search_phone('ฝ่าย IT')         # Search by department")
    print("   data = search_phone('สาขา RY')         # Search by site")


def search_phone(query: str) -> List[Dict[str, Any]]:
    """
    Simple function to search phone directory - returns raw data
    
    Args:
        query: Natural language search query (e.g., "ขอเบอร์ทองสร้อย", "701", "ฝ่าย IT")
    
    Returns:
        List[Dict]: List of matching records (raw data)
        
    Examples:
        >>> data = search_phone("ขอเบอร์ทองสร้อย")
        >>> data = search_phone("701") 
        >>> data = search_phone("ฝ่าย IT")
    """
    try:
        # Default paths
        excel_path = "excel\\all_phone.xlsx"
        json_backup = "phone_directory_with_empty.json"
        
        # Initialize search engine
        search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
        
        if not search_engine.data_records:
            return []
        
        # Parse the query to determine search type
        search_type, search_value = search_engine.parse_natural_language(query)
        
        # Perform search based on type and return raw results
        if search_type == 'phone':
            return search_engine.search_by_phone(search_value)
        elif search_type == 'name':
            return search_engine.search_by_name(search_value)
        elif search_type == 'department':
            return search_engine.search_by_department(search_value)
        elif search_type == 'site':
            return search_engine.search_by_site(search_value)
        else:
            return search_engine.search_by_name(search_value)  # Default to name search
            
    except Exception as e:
        print(f"Error in search_phone: {e}")
        return []


if __name__ == "__main__":
    print("Phone Directory Exact Match Search")
    print("=" * 40)
    print("1. Run demo")
    print("2. Interactive search (old style)")
    print("3. Smart CLI with Natural Language")
    print("4. Test LLM Integration Functions")
    print("5. Quick Test search_phone() function (NEW!)")
    
    choice = input("Choose option (1, 2, 3, 4, or 5): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        interactive_search()
    elif choice == "3":
        # New natural language CLI
        excel_path = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\all_phone.xlsx"
        json_backup = "c:\\Users\\teety\\Desktop\\NCR_CHATBOT\\phone_directory_with_empty.json"
        
        search_engine = PhoneDirectoryExactSearch(excel_path, json_backup)
        
        if not search_engine.data_records:
            print("❌ No data loaded. Please check file paths.")
        else:
            search_engine.interactive_cli()
    elif choice == "4":
        test_llm_functions()
    elif choice == "5":
        # Quick test of the simple search_phone function
        print("🚀 Quick Test: search_phone() function")
        print("=" * 40)
        
        test_queries = [
            "ขอเบอร์ทองสร้อย",
            "701", 
            "ฝ่ายธุรการ",
            "สาขา RY"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: search_phone('{query}')")
            data = search_phone(query)
            print(f"   ✅ Found {len(data)} records")
            
            if data:
                # Show first result
                first = data[0]
                name = f"{first.get('ชื่อ', '')} {first.get('นามสกุล', '')}"
                phone = first.get('เบอร์โทร', 'N/A')
                dept = first.get('ฝ่าย', 'N/A')
                print(f"   📋 Sample: {name} - เบอร์: {phone} - ฝ่าย: {dept}")
        
        print("\n💡 Usage in your LLM code:")
        print("   data = search_phone('ขอเบอร์ชื่อสมชาย')")
        print("   if data:")
        print("       phone = data[0]['เบอร์โทร']")
        print("       name = data[0]['ชื่อ']")
    else:
        print("Running demo by default...")
        main()

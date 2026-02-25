import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Union, Optional

import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Union, Optional

def load_po_data(excel_file: str = "excel/com_list.xlsx", sheet_name: str = 'Sheet1') -> pd.DataFrame:
    """
    Load PO data from Excel file or create sample data
    
    Args:
        excel_file: Path to Excel file (default: "excel/com_list.xlsx")
        sheet_name: Sheet name in Excel file
        
    Returns:
        DataFrame with PO data
    """
    try:
        print(f"📂 Loading data from: {excel_file}")
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Display basic info about loaded data
        print(f"✅ Loaded {len(df)} records from {sheet_name}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Handle different possible column names for your Excel file
        column_mapping = {
            'Po_Number': ['Po Number', 'Po_Number', 'PO Number', 'PO_Number'],
            'PO_Date': ['PO_Date', 'PO Date', 'Po_Date', 'Po Date', 'Date']
        }
        
        # Standardize column names
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col in possible_names:
                    df = df.rename(columns={col: standard_name})
                    print(f"🔄 Renamed column '{col}' to '{standard_name}'")
                    break
        
        # Convert PO_Date to datetime
        if 'PO_Date' in df.columns:
            print("🔄 Converting PO_Date to datetime...")
            df['PO_Date'] = pd.to_datetime(df['PO_Date'], errors='coerce')
            
            # Check for invalid dates
            invalid_dates = df['PO_Date'].isna().sum()
            if invalid_dates > 0:
                print(f"⚠️ Found {invalid_dates} invalid dates")
            
            # Calculate years from PO_Date to now
            print("📊 Calculating years since PO...")
            df['Years_Since_PO'] = df['PO_Date'].apply(calculate_years_from_now)
            
            # Calculate formatted age (like "2 years 4 months")
            print("📊 Calculating formatted age...")
            df['Age_Formatted'] = df['PO_Date'].apply(calculate_age_formatted)
        else:
            print("⚠️ PO_Date column not found, cannot calculate years")
            df['Years_Since_PO'] = 0.0
        
        # Display sample data
        print("\n📋 Sample data (first 3 rows):")
        print(df.head(3).to_string())
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File not found: {excel_file}")
        print("💡 Please make sure 'com_list.xlsx' is in the excel/ directory")
        print("🔄 Using sample data instead...")
        return create_sample_data()
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        print("🔄 Using sample data instead...")
        return create_sample_data()

def create_sample_data() -> pd.DataFrame:
    """Create sample data based on your example"""
    data = {
        'Client': ['NBK031', 'NBK040', 'NBK047', 'NBK052', 'NBK055', 'NBK056', 'NBK058', 'NBK059', 'NBK060'],
        'User': ['Engineer PLC', 'MBH Center', 'QA Center', 'CWS', 'Sarayut', 'Jeenaphat', 'SU', 'VN', 'WC'],
        'Po_Number': ['053665_018', '078816_018', '089428_018', '091733_018', '099699_018', '099699_018', '099696_018', '099696_018', '105719_018'],
        'PO_Date': ['4/4/2013', '3/29/2017', '9/21/2018', '1/24/2019', '2/28/2020', '2/28/2020', '2/28/2020', '2/28/2020', '1/26/2021']
    }
    df = pd.DataFrame(data)
    df['PO_Date'] = pd.to_datetime(df['PO_Date'])
    df['Years_Since_PO'] = df['PO_Date'].apply(calculate_years_from_now)
    df['Age_Formatted'] = df['PO_Date'].apply(calculate_age_formatted)
    return df

def calculate_age_formatted(po_date: datetime) -> str:
    """
    Calculate age from PO date to current date in human-readable format
    
    Args:
        po_date: PO date as datetime
        
    Returns:
        Formatted string like "2 years 4 months" or "8 months" or "3 weeks"
    """
    # Check if the date is NaT (Not a Time) or None
    if pd.isna(po_date):
        return "0 months"
    
    try:
        from dateutil.relativedelta import relativedelta
        now = datetime.now()
        
        # Calculate the difference using relativedelta for accurate years/months
        diff = relativedelta(now, po_date)
        
        years = diff.years
        months = diff.months
        days = diff.days
        
        # Format based on age
        if years > 0:
            if months > 0:
                return f"{years} years {months} months"
            else:
                return f"{years} years"
        elif months > 0:
            if days > 15:  # Round up if more than half a month
                months += 1
            return f"{months} months"
        elif days >= 7:
            weeks = days // 7
            return f"{weeks} weeks"
        else:
            return f"{days} days"
            
    except ImportError:
        # Fallback if dateutil is not available
        try:
            now = datetime.now()
            total_days = (now - po_date).days
            
            # Approximate calculation
            years = total_days // 365
            remaining_days = total_days % 365
            months = remaining_days // 30
            
            if years > 0:
                if months > 0:
                    return f"{years} years {months} months"
                else:
                    return f"{years} years"
            elif months > 0:
                return f"{months} months"
            elif total_days >= 7:
                weeks = total_days // 7
                return f"{weeks} weeks"
            else:
                return f"{total_days} days"
                
        except (TypeError, AttributeError):
            return "0 months"
    except (TypeError, AttributeError):
        return "0 months"

def calculate_years_from_now(po_date: datetime) -> float:
    """
    Calculate years from PO date to current date (for backward compatibility)
     
    Args:
        po_date: PO date as datetime
        
    Returns:
        Years as float (rounded to 2 decimal places)
    """
    # Check if the date is NaT (Not a Time) or None
    if pd.isna(po_date):
        return 0.0
    
    try:
        now = datetime.now()
        years = (now - po_date).days / 365.25  # Use 365.25 to account for leap years
        return round(years, 2)
    except (TypeError, AttributeError):
        return 0.0

def search_po_records(query: str, df: pd.DataFrame = None, search_type: str = 'auto') -> List[Dict]:
    """
    Search PO records by client, user, or PO number
    
    Args:
        query: Search term (client code, user name, or PO number)
        df: DataFrame with PO data (if None, will load sample data)
        search_type: 'auto', 'client', 'user', or 'po_number'
        
    Returns:
        List of matching records with years calculation
    """
    if df is None:
        df = load_po_data()
    
    if df.empty:
        return []
    
    query = query.strip()
    results = []
    
    if search_type == 'auto':
        # Auto-detect search type
        if query.upper().startswith('NBK'):
            search_type = 'client'
        elif '_' in query and query.replace('_', '').isdigit():
            search_type = 'po_number'
        else:
            search_type = 'user'
    
    # Perform search based on type
    try:
        if search_type == 'client':
            matches = df[df['Client'].str.contains(query, case=False, na=False)]
        elif search_type == 'user':
            matches = df[df['User'].str.contains(query, case=False, na=False)]
        elif search_type == 'po_number':
            matches = df[df['Po_Number'].str.contains(query, case=False, na=False)]
        else:
            # Search all columns
            mask = (df['Client'].str.contains(query, case=False, na=False) |
                    df['User'].str.contains(query, case=False, na=False) |
                    df['Po_Number'].str.contains(query, case=False, na=False))
            matches = df[mask]
    except KeyError as e:
        print(f"❌ Column not found: {e}")
        return []
    
    # Convert to list of dictionaries
    for _, row in matches.iterrows():
        # Handle date formatting with proper NaT check
        if pd.notna(row['PO_Date']):
            try:
                formatted_date = row['PO_Date'].strftime('%m/%d/%Y')
                # Calculate formatted age (like "2 years 4 months")
                age_formatted = calculate_age_formatted(row['PO_Date'])
            except (AttributeError, ValueError):
                formatted_date = 'Invalid Date'
                age_formatted = 'Invalid Date'
        else:
            formatted_date = 'Invalid Date'
            age_formatted = 'Invalid Date'
        
        # Handle Years_Since_PO with NaN check (keep for backward compatibility)
        years_since = row.get('Years_Since_PO', 0.0)
        if pd.isna(years_since):
            years_since = 0.0
        
        results.append({
            'Client': row['Client'],
            'User': row['User'],
            'Po_Number': row['Po_Number'],
            'PO_Date': formatted_date,
            'Years_Since_PO': years_since,
            'Age_Formatted': age_formatted,
            'Search_Type': search_type
        })
    
    return results

def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string
    """
    if not results:
        return "❌ No records found"
    
    output = f"✅ Found {len(results)} record(s):\n"
    output += "=" * 80 + "\n"
    
    for i, record in enumerate(results, 1):
        output += f"{i}. Client: {record['Client']}\n"
        output += f"   User: {record['User']}\n"
        output += f"   PO Number: {record['Po_Number']}\n"
        output += f"   PO Date: {record['PO_Date']}\n"
        # Use formatted age if available, otherwise fall back to years
        age_display = record.get('Age_Formatted', f"{record.get('Years_Since_PO', 0.0)} years")
        output += f"   Age: {age_display}\n"
        output += "-" * 50 + "\n"
    
    return output

def get_po_summary(df: pd.DataFrame = None) -> Dict:
    """
    Get summary statistics of PO data
    
    Args:
        df: DataFrame with PO data
        
    Returns:
        Summary statistics
    """
    if df is None:
        df = load_po_data()
    
    # Filter out invalid dates for summary
    valid_dates = df[pd.notna(df['PO_Date'])]
    
    summary = {
        'total_records': len(df),
        'valid_dates': len(valid_dates),
        'invalid_dates': len(df) - len(valid_dates)
    }
    
    if len(valid_dates) > 0:
        try:
            summary['date_range'] = {
                'earliest': valid_dates['PO_Date'].min().strftime('%m/%d/%Y'),
                'latest': valid_dates['PO_Date'].max().strftime('%m/%d/%Y')
            }
            summary['years_range'] = {
                'min': valid_dates['Years_Since_PO'].min(),
                'max': valid_dates['Years_Since_PO'].max(),
                'average': round(valid_dates['Years_Since_PO'].mean(), 2)
            }
        except (AttributeError, ValueError):
            summary['date_range'] = {'error': 'Error formatting dates'}
            summary['years_range'] = {'error': 'Error calculating years'}
    else:
        summary['date_range'] = {'error': 'No valid dates found'}
        summary['years_range'] = {'error': 'No valid dates found'}
    
    summary['unique_clients'] = df['Client'].nunique()
    summary['unique_users'] = df['User'].nunique()
    
    return summary

# Simple function for easy use with LLM
def search_po(query: str, excel_file: str = "excel/com_list.xlsx") -> str:
    """
    Simple function to search PO records from com_list.xlsx
    
    Args:
        query: Search term (client, user, or PO number)
        excel_file: Path to Excel file (default: "excel/com_list.xlsx")
        
    Returns:
        Formatted search results
    """
    try:
        df = load_po_data(excel_file)
        if df.empty:
            return "❌ No data loaded from Excel file"
        
        results = search_po_records(query, df)
        return format_search_results(results)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Alternative function that returns raw data for LLM processing
def search_po_data(query: str, excel_file: str = "excel/com_list.xlsx") -> List[Dict]:
    """
    Search PO records and return raw data (for LLM integration)
    
    Args:
        query: Search term (client, user, or PO number)
        excel_file: Path to Excel file (default: "excel/com_list.xlsx")
        
    Returns:
        List of dictionaries with matching records
    """
    try:
        df = load_po_data(excel_file)
        if df.empty:
            return []
        
        results = search_po_records(query, df)
        return results
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def search_po_json(query: str, excel_file: str = "excel/com_list.xlsx") -> str:
    """
    Search PO records and return results as JSON string
    
    Args:
        query: Search term (client, user, or PO number)
        excel_file: Path to Excel file (default: "excel/com_list.xlsx")
        
    Returns:
        JSON string with search results
    """
    import json
    
    try:
        df = load_po_data(excel_file)
        if df.empty:
            return json.dumps({"error": "No data loaded from Excel file", "results": []}, indent=2)
        
        results = search_po_records(query, df)
        
        # Create a structured JSON response
        response = {
            "query": query,
            "total_results": len(results),
            "search_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "query": query,
            "results": []
        }
        return json.dumps(error_response, indent=2)

def get_excel_info(excel_file: str = "excel/com_list.xlsx") -> Dict:
    """
    Get information about Excel file structure
    
    Args:
        excel_file: Path to Excel file
        
    Returns:
        Information about the Excel file
    """
    try:
        # Get sheet names
        excel_file_obj = pd.ExcelFile(excel_file)
        sheet_names = excel_file_obj.sheet_names
        
        info = {
            'file_path': excel_file,
            'sheet_names': sheet_names,
            'sheets_info': {}
        }
        
        # Get info for each sheet
        for sheet in sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet, nrows=5)  # Read only first 5 rows
                info['sheets_info'][sheet] = {
                    'columns': list(df.columns),
                    'row_count': len(df),
                    'sample_data': df.head(2).to_dict('records')
                }
            except Exception as e:
                info['sheets_info'][sheet] = {'error': str(e)}
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

# Demo function
def demo_po_search():
    """Demonstrate the PO search functionality with com_list.xlsx"""
    print("🔍 PO Search System Demo - com_list.xlsx")
    print("=" * 50)
    
    # Load data from com_list.xlsx
    df = load_po_data("excel/com_list.xlsx")
    
    if df.empty:
        print("❌ No data loaded")
        return
    
    # Show summary if data loaded successfully
    if len(df) > 0:
        summary = get_po_summary(df)
        print(f"\n📊 Summary: {summary['total_records']} records")
        print(f"📊 Valid dates: {summary['valid_dates']}, Invalid dates: {summary['invalid_dates']}")
        
        if 'date_range' in summary and 'error' not in summary['date_range']:
            print(f"📅 Date Range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
            print(f"⏱️  Years Range: {summary['years_range']['min']} to {summary['years_range']['max']} years")
        print()
    
    # Test searches with actual data
    print("🎯 Sample Searches:")
    sample_queries = []
    
    # Get some sample values from the actual data
    if 'Client' in df.columns and len(df) > 0:
        sample_client = df['Client'].iloc[0] if not df['Client'].isna().all() else None
        if sample_client:
            sample_queries.append(sample_client)
    
    if 'User' in df.columns and len(df) > 0:
        sample_user = df['User'].iloc[0] if not df['User'].isna().all() else None
        if sample_user:
            sample_queries.append(str(sample_user)[:10])  # First 10 chars
    
    # Add some default searches
    if not sample_queries:
        sample_queries = ['NBK', 'Engineer', 'QA']
    
    for query in sample_queries[:3]:  # Limit to 3 searches
        print(f"🔍 Searching for: '{query}'")
        results = search_po_records(query, df)
        if results:
            print(format_search_results(results))
        else:
            print("❌ No results found\n")

def test_search_functions():
    """Test the search functions with different queries"""
    print("🧪 Testing Search Functions")
    print("=" * 40)
    
    test_queries = ['NBK', 'Engineer', 'QA', '053665']
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        # Test formatted output
        result = search_po(query)
        print("Formatted output:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Test raw data
        data = search_po_data(query)
        print(f"Raw data: {len(data)} records found")
        
        # Test JSON output
        json_result = search_po_json(query)
        print("JSON output:")
        print(json_result[:300] + "..." if len(json_result) > 300 else json_result)

if __name__ == "__main__":
    # Check if Excel file exists
    import os
    
    excel_path = "excel/com_list.xlsx"
    if os.path.exists(excel_path):
        print("✅ Found excel/com_list.xlsx")
        demo_po_search()
    else:
        print("❌ excel/com_list.xlsx not found")
        print("📁 Current directory contents:")
        files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
        if files:
            print(f"Found these data files: {files}")
        else:
            print("No Excel/CSV files found")
        print("\n🔄 Running demo with sample data...")
        demo_po_search()
    
    # Interactive mode
    print("\n🎯 Interactive Search Mode")
    print("Enter search terms (client, user, or PO number) or 'exit' to quit")
    print("Commands: 'test' - run tests, 'info' - show Excel info, 'json <query>' - get JSON result")
    
    while True:
        query = input("\n🔍 Search: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'test':
            test_search_functions()
        elif query.lower() == 'info':
            info = get_excel_info()
            print(f"📊 Excel Info: {info}")
        elif query.lower().startswith('json '):
            # Extract the actual query after 'json '
            actual_query = query[5:].strip()
            if actual_query:
                json_result = search_po_json(actual_query)
                print("JSON Result:")
                print(json_result)
            else:
                print("⚠️ Please provide a search term after 'json'")
        elif query:
            result = search_po(query)
            print(result)
        else:
            print("⚠️ Please enter a search term")


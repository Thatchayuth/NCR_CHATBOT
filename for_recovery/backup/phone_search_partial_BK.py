import pandas as pd

excel_path = "excel\\all_phone.xlsx"

def search_phone(keyword):
    """
    Search for a keyword across all fields in the Excel file
    Returns a list of matching records (dictionaries)
    """
    # Read the Excel file
    print(keyword)
    print('#'*80)
    df = pd.read_excel(excel_path)
    
    # Convert keyword to string and make it case-insensitive
    keyword = str(keyword).lower()
    
    # Convert DataFrame to list of dictionaries (records)
    data_records = df.to_dict('records')
    results = []
    
    # Search through each record
    for record in data_records:
        # Check if keyword exists in any field of the record
        found_match = False
        for key, value in record.items():
            if str(key) =='Email' or str(key) == 'นามสกุล':
                continue
            if str(value).lower().find(keyword) != -1:  # Case-insensitive partial match
                found_match = True
                break
        
        if found_match:
            results.append(record)
    
    if len(results) > 0:
        print(f"Found {len(results)} matching rows for keyword '{keyword}':")
        print("="*80)
        for i, record in enumerate(results):
            print(f"Record {i + 1}:")
            for key, value in record.items():
                if pd.notna(value) and str(value).strip():  # Only show non-empty values
                    print(f"  {key}: {value}")
            print("-"*40)
        return results
    else:
        print(f"No matches found for keyword '{keyword}'")
        return []

def search_phone_specific(phone):
    """
    Search specifically in phone number fields with exact matching
    """
    df = pd.read_excel(excel_path)
    phone_str = str(phone).strip()
    
    # Create boolean mask for exact matching
    results = []
    
    for idx, row in df.iterrows():
        # Check main phone number (exact match)
        if str(row.get('เบอร์โทร', '')).strip() == phone_str:
            results.append(idx)
        # Check emergency phone number (exact match)  
        elif str(row.get('เบอร์ด่วน', '')).strip() == phone_str:
            results.append(idx)
    
    # Return matching rows
    if results:
        return df.loc[results]
    else:
        return pd.DataFrame()

def search_phone_exact(phone):
    """
    Search for exact phone number match using record iteration approach
    Returns a list of matching records (dictionaries)
    """
    df = pd.read_excel(excel_path)
    phone_str = str(phone).strip()
    results = []
    
    # Convert DataFrame to list of dictionaries (records)
    data_records = df.to_dict('records')
    
    for record in data_records:
        # Check main phone number
        if str(record.get('เบอร์โทร', '')).strip() == phone_str:
            results.append(record)
        elif str(record.get('เบอร์ด่วน', '')).strip() == phone_str:
            results.append(record)
        elif str(record.get('Email', '')).strip() == phone_str:
            results.append(record)
    
    return results

# Example usage:
if __name__ == "__main__":
    # Search for any keyword across all fields
    print("=== Search Examples ===")
    
    # Example 1: Search for a name (partial match)
    search_phone("มิว")
    
    # Example 2: Search for a department (partial match)
    search_phone("ธุรการ")
    
    # Example 3: Search for a phone number (partial match)
    search_phone("701")
    
    # Example 4: Search for exact phone number match
    print("\n=== Exact Phone Search Examples ===")
    exact_result = search_phone_exact("701")
    if exact_result:
        print(f"Found {len(exact_result)} exact match(es) for phone '701':")
        for i, record in enumerate(exact_result):
            print(f"Record {i+1}:")
            for key, value in record.items():
                if pd.notna(value) and str(value).strip():  # Only show non-empty values
                    print(f"  {key}: {value}")
            print("-"*40)
    else:
        print("No exact match found for phone '701'")
    
    # Example 5: Search for a site
    search_phone("ดับเพลิง")
    
    search_phone('ak')
    print('')
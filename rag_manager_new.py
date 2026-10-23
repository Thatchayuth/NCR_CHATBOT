import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import json
import fitz  # PyMuPDF
import easyocr
import os
from PIL import Image, ImageTk
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import threading

class VectorDBManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Database Manager")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.embedding = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B", 
            model_kwargs={"device": "cuda"}
        )
        self.db = None
        self.persist_dir = "last_persist_db_qwen_ollama_rev3"
        self.ocr_reader = None
        self.metadata_entries = {}
        self.search_results = {}  # Store search results for details view
        self.search_results_ids = {}  # Store document IDs for editing
        
        # Connect to existing database
        self.connect_db()
        
        # Create GUI
        self.create_gui()
        
    def connect_db(self):
        """เชื่อมต่อกับ Vector Database"""
        try:
            self.db = Chroma(
                persist_directory=self.persist_dir, 
                embedding_function=self.embedding
            )
            print("✅ เชื่อมต่อ Vector Database สำเร็จ")
            # ทดสอบการเชื่อมต่อด้วยการดึงข้อมูล
            test_docs = self.db.get()
            print(f"✅ ทดสอบการเชื่อมต่อสำเร็จ - พบ {len(test_docs['ids']) if test_docs and 'ids' in test_docs else 0} เอกสาร")
        except Exception as e:
            print(f"❌ ไม่สามารถเชื่อมต่อ Database: {e}")
            self.db = None
            
    def create_gui(self):
        """สร้าง GUI หลัก"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel - File upload and metadata
        left_frame = ttk.LabelFrame(main_frame, text="📁 Upload & Metadata", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel - Search and results
        right_frame = ttk.LabelFrame(main_frame, text="🔍 Search & Results", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure weights
        main_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
    def create_left_panel(self, parent):
        """สร้างแผงด้านซ้าย - Upload และ Metadata"""
        
        # File Upload Section
        upload_frame = ttk.LabelFrame(parent, text="📤 File Upload", padding="5")
        upload_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        upload_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(upload_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(upload_frame, textvariable=self.file_path_var, state="readonly")
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # File type detection
        self.file_type_var = tk.StringVar(value="ไม่ได้เลือกไฟล์")
        ttk.Label(upload_frame, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Label(upload_frame, textvariable=self.file_type_var).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Content Preview
        preview_frame = ttk.LabelFrame(parent, text="📋 Content Preview", padding="5")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        self.content_text = scrolledtext.ScrolledText(preview_frame, height=8, wrap=tk.WORD)
        self.content_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Metadata Section
        self.metadata_frame = ttk.LabelFrame(parent, text="📝 Metadata", padding="5")
        self.metadata_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.metadata_frame.columnconfigure(1, weight=1)
        
        # Default metadata fields
        self.create_metadata_fields()
        
        # Dynamic metadata controls
        dynamic_frame = ttk.Frame(self.metadata_frame)
        dynamic_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        dynamic_frame.columnconfigure(1, weight=1)
        
        ttk.Button(dynamic_frame, text="➕ Add Field", command=self.add_metadata_field).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(dynamic_frame, text="➖ Remove Field", command=self.remove_metadata_field).grid(row=0, column=1, padx=(0, 5))
        
        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        
        ttk.Button(action_frame, text="💾 Add to Database", command=self.add_to_database).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(action_frame, text="🔄 Clear All", command=self.clear_all).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Database info
        info_frame = ttk.LabelFrame(parent, text="📊 Database Info", padding="5")
        info_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        info_frame.columnconfigure(0, weight=1)
        
        self.db_info_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.db_info_var).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Reconnect button
        ttk.Button(info_frame, text="🔄 Reconnect Database", command=self.reconnect_database).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.update_db_info()
        
    def create_right_panel(self, parent):
        """สร้างแผงด้านขวา - Search และ Results"""
        
        # Search Section
        search_frame = ttk.LabelFrame(parent, text="🔍 Search", padding="5")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Query:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        search_entry.bind('<Return>', lambda e: self.search_database())
        
        ttk.Button(search_frame, text="Search", command=self.search_database).grid(row=0, column=2)
        
        # Search options
        options_frame = ttk.Frame(search_frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(options_frame, text="Results:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.search_limit_var = tk.StringVar(value="5")
        limit_spinbox = ttk.Spinbox(options_frame, from_=1, to=50, width=5, textvariable=self.search_limit_var)
        limit_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        self.partial_search_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Partial Search", variable=self.partial_search_var).grid(row=0, column=2, sticky=tk.W)
        
        # Results Section
        results_frame = ttk.LabelFrame(parent, text="📋 Search Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results tree
        columns = ("score", "topic", "type", "content")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="tree headings", height=15)
        
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("score", text="Score")
        self.results_tree.heading("topic", text="Topic")
        self.results_tree.heading("type", text="Type") 
        self.results_tree.heading("content", text="Content Preview")
        
        self.results_tree.column("#0", width=100)
        self.results_tree.column("score", width=80)
        self.results_tree.column("topic", width=200)
        self.results_tree.column("type", width=100)
        self.results_tree.column("content", width=300)
        
        # Scrollbars for results
        v_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Bind double-click to show details
        self.results_tree.bind("<Double-1>", self.show_result_details)
        
        # Results actions
        results_actions = ttk.Frame(results_frame)
        results_actions.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # First row of buttons
        button_row1 = ttk.Frame(results_actions)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(button_row1, text="📝 View Details", command=self.show_result_details).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="🗑️ Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="🔄 Refresh Database", command=self.update_db_info).pack(side=tk.LEFT, padx=(0, 5))
        
        # Second row of buttons
        button_row2 = ttk.Frame(results_actions)
        button_row2.pack(fill=tk.X)
        
        ttk.Button(button_row2, text="⚠️ Clear All Database", command=self.clear_all_database).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row2, text="📊 Database Stats", command=self.show_database_stats).pack(side=tk.LEFT)
        
    def create_metadata_fields(self):
        """สร้างฟิลด์ metadata เริ่มต้น"""
        default_fields = [
            ("topic", "หัวข้อ:"),
            ("type", "ประเภท:"),
            ("hardware", "อุปกรณ์:"),
            ("keyword_th", "คำค้น (ไทย):"),
            ("keyword_en", "คำค้น (EN):"),
            ("extra_keywords", "คำค้นเพิ่มเติม:"),    
            ("clip", "ลิงค์ VIDEO"),
            ("url", "ลิงค์ PDF")
        ]
        
        for i, (key, label) in enumerate(default_fields):
            ttk.Label(self.metadata_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(self.metadata_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
            self.metadata_entries[key] = {'var': var, 'label': label, 'row': i}
            
    def add_metadata_field(self):
        """เพิ่มฟิลด์ metadata ใหม่"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Metadata Field")
        dialog.geometry("300x150")
        dialog.grab_set()
        dialog.transient(self.root)
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        ttk.Label(dialog, text="Field Name:").pack(pady=5)
        field_var = tk.StringVar()
        field_entry = ttk.Entry(dialog, textvariable=field_var)
        field_entry.pack(pady=5, padx=20, fill=tk.X)
        field_entry.focus()
        
        ttk.Label(dialog, text="Label:").pack(pady=5)
        label_var = tk.StringVar()
        label_entry = ttk.Entry(dialog, textvariable=label_var)
        label_entry.pack(pady=5, padx=20, fill=tk.X)
        
        def add_field():
            field = field_var.get().strip()
            label = label_var.get().strip()
            if field and label:
                if field not in self.metadata_entries:
                    row = len(self.metadata_entries)
                    
                    label_widget = ttk.Label(self.metadata_frame, text=f"{label}:")
                    label_widget.grid(row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                    
                    var = tk.StringVar()
                    entry = ttk.Entry(self.metadata_frame, textvariable=var)
                    entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
                    
                    self.metadata_entries[field] = {
                        'var': var, 
                        'label': label, 
                        'row': row,
                        'label_widget': label_widget,
                        'entry_widget': entry
                    }
                    
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Added field: {field}")
                else:
                    messagebox.showwarning("Warning", "Field already exists!")
            else:
                messagebox.showwarning("Warning", "Please fill all fields!")
        
        def cancel():
            dialog.destroy()
            
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Add", command=add_field).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to add field
        dialog.bind('<Return>', lambda e: add_field())
        dialog.bind('<Escape>', lambda e: cancel())
        
    def remove_metadata_field(self):
        """ลบฟิลด์ metadata"""
        if len(self.metadata_entries) <= 1:
            messagebox.showwarning("Warning", "Cannot remove all fields!")
            return
            
        fields = list(self.metadata_entries.keys())
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Metadata Field")
        dialog.geometry("250x300")
        dialog.grab_set()
        dialog.transient(self.root)
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        ttk.Label(dialog, text="Select field to remove:").pack(pady=5)
        
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        listbox = tk.Listbox(listbox_frame)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for field in fields:
            listbox.insert(tk.END, f"{field} ({self.metadata_entries[field]['label']})")
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def remove_field():
            selection = listbox.curselection()
            if selection:
                field = fields[selection[0]]
                if field in self.metadata_entries:
                    # Remove widgets
                    entry_data = self.metadata_entries[field]
                    if 'label_widget' in entry_data:
                        entry_data['label_widget'].destroy()
                    if 'entry_widget' in entry_data:
                        entry_data['entry_widget'].destroy()
                    
                    del self.metadata_entries[field]
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Removed field: {field}")
                    
                    # Reorganize remaining fields
                    self.reorganize_metadata_fields()
            else:
                messagebox.showwarning("Warning", "Please select a field!")
        
        def cancel():
            dialog.destroy()
            
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Remove", command=remove_field).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        # Bind double-click to remove
        listbox.bind('<Double-1>', lambda e: remove_field())
        dialog.bind('<Escape>', lambda e: cancel())
        
    def reorganize_metadata_fields(self):
        """จัดระเบียบฟิลด์ metadata ใหม่"""
        for i, (field, data) in enumerate(self.metadata_entries.items()):
            if 'label_widget' in data and 'entry_widget' in data:
                data['label_widget'].grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                data['entry_widget'].grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
                data['row'] = i
        
    def browse_file(self):
        """เลือกไฟล์"""
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All Supported", "*.txt *.pdf *.png *.jpg *.jpeg"),
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Image files", "*.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.detect_file_type(file_path)
            self.extract_content(file_path)
            
    def detect_file_type(self, file_path):
        """ตรวจสอบประเภทไฟล์"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            self.file_type_var.set("📄 Text File")
        elif ext == '.pdf':
            self.file_type_var.set("📕 PDF File")
        elif ext in ['.png', '.jpg', '.jpeg']:
            self.file_type_var.set("🖼️ Image File")
        else:
            self.file_type_var.set("❓ Unknown File Type")
            
    def extract_content(self, file_path):
        """แยกเนื้อหาจากไฟล์"""
        self.content_text.delete(1.0, tk.END)
        self.content_text.insert(tk.END, "🔄 กำลังประมวลผล...\n")
        self.root.update()
        
        def process_file():
            try:
                content = ""
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                elif ext == '.pdf':
                    content = self.extract_pdf_content(file_path)
                    
                elif ext in ['.png', '.jpg', '.jpeg']:
                    content = self.extract_image_content(file_path)
                    
                # Update GUI in main thread
                self.root.after(0, lambda: self.update_content_display(content))
                
            except Exception as e:
                error_msg = f"❌ Error processing file: {str(e)}"
                self.root.after(0, lambda: self.update_content_display(error_msg))
                
        # Run in background thread
        threading.Thread(target=process_file, daemon=True).start()
        
    def extract_pdf_content(self, file_path):
        """แยกเนื้อหาจาก PDF"""
        content = ""
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try text extraction first
                text = page.get_text()
                if text.strip():
                    content += f"\n--- หน้า {page_num + 1} ---\n{text}\n"
                else:
                    # If no text, try OCR on images
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Save temp image for OCR
                    temp_path = "temp_page.png"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)
                    
                    ocr_text = self.extract_image_content(temp_path)
                    content += f"\n--- หน้า {page_num + 1} (OCR) ---\n{ocr_text}\n"
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            doc.close()
            
        except Exception as e:
            content = f"❌ ไม่สามารถอ่าน PDF: {str(e)}"
            
        return content
        
    def extract_image_content(self, file_path):
        """แยกเนื้อหาจากรูปภาพด้วย OCR"""
        try:
            if self.ocr_reader is None:
                self.ocr_reader = easyocr.Reader(['th', 'en'])
                
            results = self.ocr_reader.readtext(file_path)
            content = ""
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    content += f"{text}\n"
                    
            return content if content else "❌ ไม่พบข้อความในรูปภาพ"
            
        except Exception as e:
            return f"❌ ไม่สามารถอ่านรูปภาพ: {str(e)}"
            
    def update_content_display(self, content):
        """อัปเดตการแสดงเนื้อหา"""
        self.content_text.delete(1.0, tk.END)
        self.content_text.insert(tk.END, content)
        
    def add_to_database(self):
        """เพิ่มข้อมูลเข้า Vector Database"""
        original_content = self.content_text.get(1.0, tk.END).strip()
        if not original_content or original_content == "🔄 กำลังประมวลผล...":
            messagebox.showwarning("Warning", "ไม่มีเนื้อหาให้เพิ่ม!")
            return

        # Collect metadata
        metadata = {}

        for key, data in self.metadata_entries.items():
            value = data['var'].get().strip()
            if value:
                metadata[key] = value

        # Format page_content with topic and content structure
        topic = metadata.get('topic', 'ไม่มีหัวข้อ')
        formatted_content = f"หัวข้อ: {topic}\nเนื้อหา:\n{original_content}"

        # Create document with formatted content
        doc = Document(page_content=formatted_content, metadata=metadata)

        try:
            if self.db:
                self.db.add_documents([doc])
                self.db.persist()
                messagebox.showinfo("Success", "เพิ่มข้อมูลสำเร็จ!")
                self.update_db_info()
                self.clear_all()
            else:
                messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database!")

        except Exception as e:
            messagebox.showerror("Error", f"ไม่สามารถเพิ่มข้อมูล: {str(e)}") 
    
    def search_database(self):
        """ค้นหาใน Vector Database"""
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "กรุณาใส่คำค้นหา!")
            return

        if not self.db:
            messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database!")
            return

        try:
            # Clear previous results and stored data
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            self.search_results.clear()
            self.search_results_ids.clear()

            limit = int(self.search_limit_var.get())

            # Get all database IDs for matching
            all_data = self.db.get()
            all_ids = all_data['ids']
            all_docs = all_data['documents']
            all_metadatas = all_data['metadatas']

            if self.partial_search_var.get():
                # Similarity search
                results = self.db.similarity_search_with_score(query, k=limit)

                for i, (doc, score) in enumerate(results):
                    topic = doc.metadata.get('topic', 'ไม่มีหัวข้อ')
                    doc_type = doc.metadata.get('type', '-')
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content

                    item_id = f"result_{i}"
                    self.results_tree.insert("", "end",
                                           iid=item_id,
                                           text=f"Result {i+1}",
                                           values=(f"{score:.4f}", topic, doc_type, content_preview))

                    # Store document data for details view
                    self.search_results[item_id] = doc

                    # Find and store document ID
                    doc_id = self.find_document_id(doc, all_ids, all_docs, all_metadatas)
                    if doc_id:
                        self.search_results_ids[item_id] = doc_id
            else:
                # Exact search
                results = self.db.similarity_search(query, k=limit)

                for i, doc in enumerate(results):
                    topic = doc.metadata.get('topic', 'ไม่มีหัวข้อ')
                    doc_type = doc.metadata.get('type', '-')
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content

                    item_id = f"result_{i}"
                    self.results_tree.insert("", "end",
                                           iid=item_id,
                                           text=f"Result {i+1}",
                                           values=("N/A", topic, doc_type, content_preview))

                    # Store document data for details view
                    self.search_results[item_id] = doc

                    # Find and store document ID
                    doc_id = self.find_document_id(doc, all_ids, all_docs, all_metadatas)
                    if doc_id:
                        self.search_results_ids[item_id] = doc_id

        except Exception as e:
            messagebox.showerror("Error", f"ไม่สามารถค้นหา: {str(e)}")

    def find_document_id(self, doc, all_ids, all_docs, all_metadatas):
        """หา document ID จากการเปรียบเทียบ content และ metadata"""
        try:
            for i, (stored_doc, stored_metadata) in enumerate(zip(all_docs, all_metadatas)):
                if stored_doc == doc.page_content and stored_metadata == doc.metadata:
                    return all_ids[i]
            return None
        except Exception as e:
            print(f"Error finding document ID: {e}")
            return None

    def update_document(self, doc_id, new_content, new_metadata):
        """อัพเดท document ใน Vector Database"""
        try:
            if not self.db or not doc_id:
                return False

            # Format page_content with topic and content structure
            topic = new_metadata.get('topic', 'ไม่มีหัวข้อ')
            formatted_content = f"หัวข้อ: {topic}\nContent:\n{new_content}"

            # Delete old document
            self.db.delete(ids=[doc_id])

            # Add new document with formatted content
            new_doc = Document(page_content=formatted_content, metadata=new_metadata)
            self.db.add_documents([new_doc])

            # Persist changes
            self.db.persist()

            return True

        except Exception as e:
            print(f"Error updating document: {e}")
            return False

    def show_result_details(self, event=None):
        """แสดงรายละเอียดของผลค้นหา"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "กรุณาเลือกรายการที่ต้องการดู!")
            return

        item_id = selection[0]
        if item_id not in self.search_results:
            messagebox.showerror("Error", "ไม่พบข้อมูลรายละเอียด!")
            return

        doc = self.search_results[item_id]
        doc_id = self.search_results_ids.get(item_id)

        # Create detail window
        detail_window = tk.Toplevel(self.root)
        detail_window.title("📄 Document Details - Edit Mode")
        detail_window.geometry("900x700")
        detail_window.transient(self.root)
        detail_window.grab_set()

        # Center the window
        detail_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 50))

        # Extract content from formatted page_content
        original_content = self.extract_content_from_formatted(doc.page_content)

        # Content Section
        content_frame = ttk.LabelFrame(detail_window, text="📄 Content (Editable)", padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        content_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, height=15)
        content_text.pack(fill=tk.BOTH, expand=True)
        content_text.insert(tk.END, original_content)

        # Metadata Section (Editable)
        metadata_frame = ttk.LabelFrame(detail_window, text="📝 Metadata (Editable)", padding="10")
        metadata_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create scrollable frame for metadata entries
        canvas = tk.Canvas(metadata_frame, height=150)
        scrollbar = ttk.Scrollbar(metadata_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create editable metadata fields
        metadata_vars = {}
        row = 0
        for key, value in doc.metadata.items():
            ttk.Label(scrollable_frame, text=f"{key}:", width=20).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=value if value else "")
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=60)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
            metadata_vars[key] = var
            row += 1

        scrollable_frame.columnconfigure(1, weight=1)

        # Add new metadata field button
        def add_new_field():
            new_key = simpledialog.askstring("New Field", "Enter field name:", parent=detail_window)
            if new_key and new_key.strip():
                new_key = new_key.strip()
                if new_key not in metadata_vars:
                    ttk.Label(scrollable_frame, text=f"{new_key}:", width=20).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    var = tk.StringVar(value="")
                    entry = ttk.Entry(scrollable_frame, textvariable=var, width=60)
                    entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                    metadata_vars[new_key] = var
                    canvas.configure(scrollregion=canvas.bbox("all"))
                else:
                    messagebox.showwarning("Warning", "Field already exists!", parent=detail_window)

        add_field_btn = ttk.Button(metadata_frame, text="➕ Add Field", command=add_new_field)
        add_field_btn.pack(side=tk.BOTTOM, pady=5)

        # Info Section
        info_frame = ttk.Frame(detail_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        if doc_id:
            ttk.Label(info_frame, text=f"Document ID: {doc_id}", foreground="gray").pack(side=tk.LEFT)
        else:
            ttk.Label(info_frame, text="⚠️ Document ID not found - Edit disabled", foreground="red").pack(side=tk.LEFT)

        # Button Section
        button_frame = ttk.Frame(detail_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def save_changes():
            if not doc_id:
                messagebox.showerror("Error", "Cannot save: Document ID not found!", parent=detail_window)
                return

            # Get new content
            new_content = content_text.get(1.0, tk.END).strip()
            if not new_content:
                messagebox.showwarning("Warning", "Content cannot be empty!", parent=detail_window)
                return

            # Get new metadata
            new_metadata = {}
            for key, var in metadata_vars.items():
                value = var.get().strip()
                if value:
                    new_metadata[key] = value

            # Update document
            if self.update_document(doc_id, new_content, new_metadata):
                messagebox.showinfo("Success", "Document updated successfully!", parent=detail_window)
                # Refresh search results
                self.search_database()
                detail_window.destroy()
            else:
                messagebox.showerror("Error", "Failed to update document!", parent=detail_window)

        def cancel():
            detail_window.destroy()

        ttk.Button(button_frame, text="💾 Save Changes", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        # Bind Escape key to close
        detail_window.bind('<Escape>', lambda _: cancel())

    def extract_content_from_formatted(self, page_content):
        """แยกเนื้อหา Content ออกจาก formatted page_content (ไม่รวมหัวข้อ)"""
        try:
            # หาตำแหน่งของ "Content:\n"
            if "Content:\n" in page_content:
                parts = page_content.split("Content:\n", 1)
                if len(parts) == 2:
                    return parts[1].strip()
            # ถ้าไม่มีรูปแบบที่คาดหวัง ให้คืนค่าทั้งหมด
            return page_content.strip()
        except Exception as e:
            print(f"Error extracting content from formatted: {e}")
            return page_content

    def show_full_content(self, content):
        """แสดง full content รวม enhanced parts"""
        view_window = tk.Toplevel(self.root)
        view_window.title("📄 Full Document Content (With Enhanced Metadata)")
        view_window.geometry("800x600")
        view_window.transient(self.root)

        # Center the window
        view_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 150, self.root.winfo_rooty() + 100))

        content_frame = ttk.Frame(view_window, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

        close_btn = ttk.Button(view_window, text="Close", command=view_window.destroy)
        close_btn.pack(pady=5)

        view_window.bind('<Escape>', lambda _: view_window.destroy())
        
    def delete_selected(self):
        """ลบรายการที่เลือก"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "กรุณาเลือกรายการที่ต้องการลบ!")
            return
            
        if not self.db:
            messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database!")
            return
        
        # ขอยืนยันการลบ
        result = messagebox.askyesno(
            "Confirm Delete", 
            f"คุณต้องการลบรายการที่เลือก {len(selection)} รายการ หรือไม่?\n\n⚠️ การลบจะไม่สามารถกู้คืนได้!",
            icon='warning'
        )
        
        if not result:
            return
            
        try:
            deleted_count = 0
            failed_items = []
            
            for item_id in selection:
                try:
                    # ดึงข้อมูลเอกสาร
                    if item_id in self.search_results:
                        doc = self.search_results[item_id]
                        
                        # หาวิธีลบเอกสารจาก ChromaDB
                        # วิธีที่ 1: ลบด้วย content matching
                        try:
                            # ค้นหาเอกสารที่ตรงกันทุกประการ
                            matching_docs = self.db.similarity_search(
                                doc.page_content, 
                                k=50  # ค้นหาเยอะๆ เพื่อให้แน่ใจว่าเจอ
                            )
                            
                            # หาเอกสารที่ตรงกันแน่นอน
                            exact_match = None
                            for match_doc in matching_docs:
                                if (match_doc.page_content == doc.page_content and 
                                    match_doc.metadata == doc.metadata):
                                    exact_match = match_doc
                                    break
                            
                            if exact_match:
                                # วิธีที่ 2: ลบโดยการดึง IDs ทั้งหมดแล้วหาตัวที่ตรงกัน
                                all_data = self.db.get()
                                
                                for i, (stored_doc, stored_metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                                    if (stored_doc == doc.page_content and 
                                        stored_metadata == doc.metadata):
                                        # ลบด้วย ID
                                        doc_id = all_data['ids'][i]
                                        self.db.delete(ids=[doc_id])
                                        deleted_count += 1
                                        
                                        # ลบจาก tree และ search results
                                        self.results_tree.delete(item_id)
                                        if item_id in self.search_results:
                                            del self.search_results[item_id]
                                        break
                                else:
                                    failed_items.append(f"ไม่พบเอกสาร ID: {item_id}")
                            else:
                                failed_items.append(f"ไม่พบเอกสารที่ตรงกัน: {item_id}")
                                
                        except Exception as delete_error:
                            failed_items.append(f"Error deleting {item_id}: {str(delete_error)}")
                    else:
                        failed_items.append(f"ไม่พบข้อมูลใน cache: {item_id}")
                        
                except Exception as item_error:
                    failed_items.append(f"Error processing {item_id}: {str(item_error)}")
            
            # บันทึกการเปลี่ยนแปลง
            if deleted_count > 0:
                self.db.persist()
                self.update_db_info()
            
            # แสดงผลลัพธ์
            if deleted_count > 0 and not failed_items:
                messagebox.showinfo("Success", f"ลบสำเร็จ {deleted_count} รายการ!")
            elif deleted_count > 0 and failed_items:
                messagebox.showwarning(
                    "Partial Success", 
                    f"ลบสำเร็จ {deleted_count} รายการ\n\nรายการที่ลบไม่สำเร็จ:\n" + "\n".join(failed_items[:5])
                )
            else:
                messagebox.showerror(
                    "Failed", 
                    f"ไม่สามารถลบรายการใดได้\n\nปัญหา:\n" + "\n".join(failed_items[:5])
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการลบ: {str(e)}")
        
    def clear_all(self):
        """ล้างข้อมูลทั้งหมด"""
        self.file_path_var.set("")
        self.file_type_var.set("ไม่ได้เลือกไฟล์")
        self.content_text.delete(1.0, tk.END)
        
        for data in self.metadata_entries.values():
            data['var'].set("")
            
    def update_db_info(self):
        """อัปเดตข้อมูล Database"""
        try:
            print(f"🔍 Checking database: self.db = {self.db}")
            print(f"🔍 Database type: {type(self.db)}")
            
            if self.db is not None:
                print("✅ Database object exists, trying to get documents...")
                # ใช้วิธีที่ปลอดภัยกว่าในการนับเอกสาร
                all_docs = self.db.get()
                print(f"📄 Retrieved docs: {type(all_docs)}")
                count = len(all_docs['ids']) if all_docs and 'ids' in all_docs else 0
                self.db_info_var.set(f"📊 Total Documents: {count}")
                print(f"📊 Database Info Updated: {count} documents")
            else:
                self.db_info_var.set("❌ Database not connected")
                print("❌ Database object is None in update_db_info")
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.db_info_var.set(error_msg)
            print(f"❌ Error in update_db_info: {e}")
    
    def reconnect_database(self):
        """เชื่อมต่อกับ Database ใหม่"""
        self.db_info_var.set("🔄 กำลังเชื่อมต่อ...")
        self.root.update()
        
        try:
            # ลองเชื่อมต่อใหม่
            self.connect_db()
            self.update_db_info()
            
            if self.db:
                messagebox.showinfo("Success", "เชื่อมต่อ Database สำเร็จ!")
            else:
                messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database ได้!")
                
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.db_info_var.set("❌ Connection failed")
    
    def clear_all_database(self):
        """ลบข้อมูลทั้งหมดใน Database"""
        if not self.db:
            messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database!")
            return
        
        # ขอยืนยันการลบ
        result = messagebox.askyesnocancel(
            "⚠️ DANGER: Clear All Database", 
            "คุณต้องการลบข้อมูลทั้งหมดใน Database หรือไม่?\n\n" +
            "⚠️ การลบจะไม่สามารถกู้คืนได้!\n" +
            "⚠️ ข้อมูลทั้งหมดจะหายไปถาวร!\n\n" +
            "Yes = ลบทั้งหมด\n" +
            "No = ยกเลิก",
            icon='warning'
        )
        
        if result is True:  # Yes clicked
            try:
                # Get all IDs
                all_data = self.db.get()
                all_ids = all_data['ids']
                
                if not all_ids:
                    messagebox.showinfo("Info", "Database ว่างเปล่าอยู่แล้ว")
                    return
                
                # Delete all documents
                self.db.delete(ids=all_ids)
                self.db.persist()
                
                # Clear search results
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)
                self.search_results.clear()
                
                # Update info
                self.update_db_info()
                
                messagebox.showinfo("Success", f"ลบข้อมูลทั้งหมดสำเร็จ! ({len(all_ids)} รายการ)")
                
            except Exception as e:
                messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการลบ: {str(e)}")
    
    def show_database_stats(self):
        """แสดงสถิติ Database"""
        if not self.db:
            messagebox.showerror("Error", "ไม่สามารถเชื่อมต่อ Database!")
            return
        
        try:
            # Get all data
            all_data = self.db.get()
            
            if not all_data or not all_data['ids']:
                messagebox.showinfo("Database Stats", "Database ว่างเปล่า")
                return
            
            total_docs = len(all_data['ids'])
            
            # Analyze metadata
            metadata_stats = {}
            total_content_length = 0
            
            for i, (doc_content, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                total_content_length += len(doc_content)
                
                if metadata:
                    for key, value in metadata.items():
                        if key not in metadata_stats:
                            metadata_stats[key] = {'count': 0, 'has_value': 0, 'unique_values': set()}
                        metadata_stats[key]['count'] += 1
                        if value and str(value).strip():
                            metadata_stats[key]['has_value'] += 1
                            metadata_stats[key]['unique_values'].add(str(value))
            
            # Create stats window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("📊 Database Statistics")
            stats_window.geometry("600x500")
            stats_window.transient(self.root)
            stats_window.grab_set()
            
            # Center the window
            stats_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 50))
            
            # Main stats
            main_frame = ttk.LabelFrame(stats_window, text="📈 Overview", padding="10")
            main_frame.pack(fill=tk.X, padx=10, pady=5)
            
            stats_text = f"""📊 Total Documents: {total_docs:,}
📄 Average Content Length: {total_content_length // total_docs:,} characters
📝 Total Content: {total_content_length:,} characters
🏷️ Metadata Fields: {len(metadata_stats)}"""
            
            ttk.Label(main_frame, text=stats_text, justify=tk.LEFT).pack(anchor=tk.W)
            
            # Metadata stats
            if metadata_stats:
                metadata_frame = ttk.LabelFrame(stats_window, text="🏷️ Metadata Statistics", padding="10")
                metadata_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                metadata_text = scrolledtext.ScrolledText(metadata_frame, height=15, wrap=tk.WORD)
                metadata_text.pack(fill=tk.BOTH, expand=True)
                
                for field, stats in metadata_stats.items():
                    unique_count = len(stats['unique_values'])
                    fill_rate = (stats['has_value'] / stats['count']) * 100
                    
                    metadata_text.insert(tk.END, 
                        f"🔹 {field}:\n" +
                        f"   - Documents: {stats['has_value']}/{stats['count']} ({fill_rate:.1f}%)\n" +
                        f"   - Unique values: {unique_count}\n" +
                        f"   - Sample values: {list(stats['unique_values'])[:5]}\n\n"
                    )
                
                metadata_text.config(state=tk.DISABLED)
            
            # Close button
            close_frame = ttk.Frame(stats_window)
            close_frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Button(close_frame, text="Close", command=stats_window.destroy).pack(side=tk.RIGHT)
            
            # Bind Escape key to close
            stats_window.bind('<Escape>', lambda e: stats_window.destroy())
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการแสดงสถิติ: {str(e)}")

def main():
    root = tk.Tk()
    app = VectorDBManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
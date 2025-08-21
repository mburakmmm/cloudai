#!/usr/bin/env python3
"""
Parquet to SQLite Converter GUI
Tkinter arayüzlü, parquet dosyalarını chatbot eğitimi için uygun formatta SQLite'a yazan araç

Özellikler:
- File picker ile dosya seçimi
- SQLite veritabanına yazma (2-cloudai.db)
- Tüm parquet.cursorrules özellikleri
- Gerçek zamanlı önizleme
- İlerleme çubuğu
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
import re
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from langdetect import detect, LangDetectException


class ParquetConverterGUI:
    """Tkinter arayüzlü parquet converter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet to SQLite Converter - CloudAI")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Veritabanı yolu - data klasörü altında
        self.db_path = "data/2-cloudai.db"
        
        # Seçilen dosya
        self.selected_file = None
        
        # Converter ayarları
        self.converter_settings = {
            'text_col': 'corpus_text',  # Varsayılan olarak corpus_text kullan
            'title_col': None,
            'body_col': None,
            'lang': None,
            'intent_prefix': None,
            'max_response_chars': None
        }
        
        self.setup_ui()
        self.create_database()
    
    def setup_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        # Ana başlık
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚀 Parquet to SQLite Converter", 
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        # Ana container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sol panel - Ayarlar
        left_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Sağ panel - Önizleme ve sonuçlar
        right_panel = tk.Frame(main_container, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Sol panel - Ayarlar"""
        # Başlık
        settings_label = tk.Label(
            parent, 
            text="⚙️ Dönüştürme Ayarları", 
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        settings_label.pack(pady=10)
        
        # Dosya seçimi
        file_frame = tk.LabelFrame(parent, text="📁 Dosya Seçimi", bg='white', padx=10, pady=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=self.file_path_var, width=40, state='readonly')
        file_entry.pack(side='left', padx=(0, 10))
        
        browse_btn = tk.Button(
            file_frame, 
            text="Dosya Seç", 
            command=self.browse_file,
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=20
        )
        browse_btn.pack(side='right')
        
        # Kolon ayarları
        columns_frame = tk.LabelFrame(parent, text="📊 Kolon Ayarları", bg='white', padx=10, pady=10)
        columns_frame.pack(fill='x', padx=10, pady=5)
        
        # Text kolonu
        tk.Label(columns_frame, text="Text Kolonu:", bg='white').grid(row=0, column=0, sticky='w', pady=2)
        self.text_col_var = tk.StringVar(value='corpus_text')  # Varsayılan olarak corpus_text
        text_col_entry = tk.Entry(columns_frame, textvariable=self.text_col_var, width=20)
        text_col_entry.grid(row=0, column=1, padx=(10, 0), pady=2)
        
        # Title kolonu
        tk.Label(columns_frame, text="Title Kolonu:", bg='white').grid(row=1, column=0, sticky='w', pady=2)
        self.title_col_var = tk.StringVar()
        title_col_entry = tk.Entry(columns_frame, textvariable=self.title_col_var, width=20)
        title_col_entry.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        # Body kolonu
        tk.Label(columns_frame, text="Body Kolonu:", bg='white').grid(row=2, column=0, sticky='w', pady=2)
        self.body_col_var = tk.StringVar()
        body_col_entry = tk.Entry(columns_frame, textvariable=self.body_col_var, width=20)
        body_col_entry.grid(row=2, column=1, padx=(10, 0), pady=2)
        
        # Diğer ayarlar
        other_frame = tk.LabelFrame(parent, text="🔧 Diğer Ayarlar", bg='white', padx=10, pady=10)
        other_frame.pack(fill='x', padx=10, pady=5)
        
        # Dil
        tk.Label(other_frame, text="Dil Kodu:", bg='white').grid(row=0, column=0, sticky='w', pady=2)
        self.lang_var = tk.StringVar()
        lang_entry = tk.Entry(other_frame, textvariable=self.lang_var, width=20)
        lang_entry.grid(row=0, column=1, padx=(10, 0), pady=2)
        # Placeholder için tooltip
        lang_entry.insert(0, "auto")
        lang_entry.bind('<FocusIn>', lambda e: lang_entry.delete(0, tk.END) if lang_entry.get() == "auto" else None)
        lang_entry.bind('<FocusOut>', lambda e: lang_entry.insert(0, "auto") if not lang_entry.get() else None)
        
        # Intent prefix
        tk.Label(other_frame, text="Intent Prefix:", bg='white').grid(row=1, column=0, sticky='w', pady=2)
        self.intent_prefix_var = tk.StringVar()
        intent_prefix_entry = tk.Entry(other_frame, textvariable=self.intent_prefix_var, width=20)
        intent_prefix_entry.grid(row=1, column=1, padx=(10, 0), pady=2)
        
        # Max response chars
        tk.Label(other_frame, text="Max Response:", bg='white').grid(row=2, column=0, sticky='w', pady=2)
        self.max_chars_var = tk.StringVar()
        max_chars_entry = tk.Entry(other_frame, textvariable=self.max_chars_var, width=20)
        max_chars_entry.grid(row=2, column=1, padx=(10, 0), pady=2)
        
        # Dönüştürme butonu
        convert_btn = tk.Button(
            parent, 
            text="🔄 Dönüştür ve Veritabanına Yaz", 
            command=self.start_conversion,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=30,
            pady=10
        )
        convert_btn.pack(pady=20)
        
        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            parent, 
            variable=self.progress_var, 
            maximum=100,
            length=300
        )
        self.progress_bar.pack(pady=10)
        
        # Durum etiketi
        self.status_var = tk.StringVar(value="Hazır")
        status_label = tk.Label(
            parent, 
            textvariable=self.status_var,
            bg='white',
            fg='#7f8c8d'
        )
        status_label.pack()
    
    def setup_right_panel(self, parent):
        """Sağ panel - Önizleme ve sonuçlar"""
        # Notebook (sekmeli arayüz)
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Önizleme sekmesi
        preview_frame = tk.Frame(notebook, bg='white')
        notebook.add(preview_frame, text="👁️ Önizleme")
        
        preview_label = tk.Label(
            preview_frame, 
            text="📊 Dosya Önizlemesi", 
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        preview_label.pack(pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, 
            width=60, 
            height=20,
            bg='#f8f9fa',
            font=('Consolas', 9)
        )
        self.preview_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sonuçlar sekmesi
        results_frame = tk.Frame(notebook, bg='white')
        notebook.add(results_frame, text="📋 Sonuçlar")
        
        results_label = tk.Label(
            results_frame, 
            text="✅ Dönüştürme Sonuçları", 
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        results_label.pack(pady=10)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            width=60, 
            height=20,
            bg='#f8f9fa',
            font=('Consolas', 9)
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Veritabanı sekmesi
        db_frame = tk.Frame(notebook, bg='white')
        notebook.add(db_frame, text="🗄️ Veritabanı")
        
        db_label = tk.Label(
            db_frame, 
            text="💾 Veritabanı Durumu", 
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        db_label.pack(pady=10)
        
        self.db_text = scrolledtext.ScrolledText(
            db_frame, 
            width=60, 
            height=20,
            bg='#f8f9fa',
            font=('Consolas', 9)
        )
        self.db_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def browse_file(self):
        """Dosya seçimi"""
        file_path = filedialog.askopenfilename(
            title="Parquet Dosyası Seç",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_path_var.set(file_path)
            self.load_preview()
    
    def load_preview(self):
        """Dosya önizlemesi yükle"""
        try:
            if not self.selected_file:
                return
            
            # Parquet dosyasını oku
            df = pd.read_parquet(self.selected_file)
            
            # Önizleme metni oluştur
            preview = f"📁 Dosya: {os.path.basename(self.selected_file)}\n"
            preview += f"📊 Toplam Satır: {len(df)}\n"
            preview += f"📋 Kolonlar: {list(df.columns)}\n\n"
            
            # İlk 5 satırı göster
            preview += "🔍 İlk 5 Satır:\n"
            preview += "=" * 50 + "\n"
            
            for i, row in df.head().iterrows():
                preview += f"Satır {i+1}:\n"
                for col in df.columns:
                    value = str(row[col])[:100]  # İlk 100 karakter
                    if len(str(row[col])) > 100:
                        value += "..."
                    preview += f"  {col}: {value}\n"
                preview += "\n"
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview)
            
            # Kolon adlarını otomatik doldur
            # Text kolonu için mevcut kolonlardan birini seç
            if 'corpus_text' in df.columns:
                self.text_col_var.set('corpus_text')
            elif 'text' in df.columns:
                self.text_col_var.set('text')
            elif 'content' in df.columns:
                self.text_col_var.set('content')
            elif 'body' in df.columns:
                self.text_col_var.set('body')
            else:
                # İlk kolonu kullan
                first_col = df.columns[0]
                self.text_col_var.set(first_col)
            
            # Title kolonu için
            if 'title' in df.columns:
                self.title_col_var.set('title')
            elif 'baslik' in df.columns:
                self.title_col_var.set('baslik')
            elif 'header' in df.columns:
                self.title_col_var.set('header')
            
            # Body kolonu için
            if 'body' in df.columns:
                self.body_col_var.set('body')
            elif 'content' in df.columns:
                self.body_col_var.set('content')
            elif 'icerik' in df.columns:
                self.body_col_var.set('icerik')
            
            self.status_var.set(f"Önizleme yüklendi: {len(df)} satır")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya önizleme hatası: {str(e)}")
            self.status_var.set("Önizleme hatası")
    
    def create_database(self):
        """Veritabanı ve tabloları oluştur"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # conversations tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent TEXT,
                    lang TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # metadata tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    total_rows INTEGER,
                    processed_rows INTEGER,
                    conversion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.update_db_status("Veritabanı oluşturuldu")
            
        except Exception as e:
            messagebox.showerror("Veritabanı Hatası", f"Veritabanı oluşturma hatası: {str(e)}")
    
    def update_db_status(self, message):
        """Veritabanı durumunu güncelle"""
        self.db_text.delete(1.0, tk.END)
        self.db_text.insert(1.0, f"🗄️ Veritabanı: {self.db_path}\n")
        self.db_text.insert(tk.END, f"📅 {message}\n\n")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tablo bilgileri
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            self.db_text.insert(tk.END, "📋 Tablolar:\n")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                self.db_text.insert(tk.END, f"  {table[0]}: {count} kayıt\n")
            
            conn.close()
            
        except Exception as e:
            self.db_text.insert(tk.END, f"❌ Hata: {str(e)}\n")
    
    def start_conversion(self):
        """Dönüştürme işlemini başlat"""
        if not self.selected_file:
            messagebox.showwarning("Uyarı", "Lütfen önce bir parquet dosyası seçin!")
            return
        
        # Ayarları topla
        self.converter_settings.update({
            'text_col': self.text_col_var.get(),
            'title_col': self.title_col_var.get() if self.title_col_var.get() else None,
            'body_col': self.body_col_var.get() if self.body_col_var.get() else None,
            'lang': self.lang_var.get() if self.lang_var.get() else None,
            'intent_prefix': self.intent_prefix_var.get() if self.intent_prefix_var.get() else None,
            'max_response_chars': int(self.max_chars_var.get()) if self.max_chars_var.get() else None
        })
        
        # Ayrı thread'de çalıştır
        conversion_thread = threading.Thread(target=self.run_conversion)
        conversion_thread.daemon = True
        conversion_thread.start()
    
    def run_conversion(self):
        """Dönüştürme işlemini çalıştır"""
        try:
            self.status_var.set("Dönüştürme başlıyor...")
            self.progress_var.set(0)
            
            # Converter oluştur
            converter = ParquetToJsonlConverter(**self.converter_settings)
            
            # Dönüştür ve veritabanına yaz
            processed_count = converter.convert_to_database(
                self.selected_file, 
                self.db_path,
                progress_callback=self.update_progress
            )
            
            # Sonuçları göster
            self.show_results(processed_count)
            self.update_db_status(f"Dönüştürme tamamlandı: {processed_count} satır")
            
        except Exception as e:
            self.status_var.set(f"Hata: {str(e)}")
            messagebox.showerror("Dönüştürme Hatası", str(e))
    
    def update_progress(self, current, total):
        """İlerleme çubuğunu güncelle"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.status_var.set(f"İşleniyor: {current}/{total}")
    
    def show_results(self, processed_count):
        """Sonuçları göster"""
        results = f"🎉 Dönüştürme Tamamlandı!\n\n"
        results += f"📁 Giriş Dosyası: {os.path.basename(self.selected_file)}\n"
        results += f"💾 Veritabanı: {self.db_path}\n"
        results += f"📝 İşlenen Satır: {processed_count}\n\n"
        
        results += "🔧 Kullanılan Ayarlar:\n"
        for key, value in self.converter_settings.items():
            results += f"  {key}: {value or 'Otomatik'}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        
        self.status_var.set(f"Dönüştürme tamamlandı: {processed_count} satır")
        self.progress_var.set(100)


class ParquetToJsonlConverter:
    """Parquet dosyalarını JSONL formatına dönüştüren sınıf"""
    
    def __init__(self, 
                 text_col: str = "text",
                 title_col: Optional[str] = None,
                 body_col: Optional[str] = None,
                 lang: Optional[str] = None,
                 intent_prefix: Optional[str] = None,
                 max_response_chars: Optional[int] = None):
        """
        Converter'ı başlat
        
        Args:
            text_col: Ham metin kolonu adı
            title_col: Başlık kolonu adı (opsiyonel)
            body_col: İçerik kolonu adı (opsiyonel)
            lang: Zorla kullanılacak dil kodu
            intent_prefix: Intent'e eklenecek ön ek
            max_response_chars: Response uzunluk sınırı
        """
        self.text_col = text_col
        self.title_col = title_col
        self.body_col = body_col
        self.lang = lang
        self.intent_prefix = intent_prefix
        self.max_response_chars = max_response_chars
        
        # Türkçe prompt şablonları
        self.tr_prompts = [
            "{title} nedir ve hangi amaçlarla kullanılır?",
            "{title} hakkında detaylı bilgi verir misin?",
            "{title} nedir? Özellikleri ve kullanım alanları nelerdir?",
            "{title} konusunda bilgi alabilir miyim?"
        ]
        
        self.en_prompts = [
            "What is {title} and what is it used for?",
            "Can you provide detailed information about {title}?",
            "What is {title}? What are its features and applications?",
            "I would like to learn about {title}."
        ]
    
    def extract_title_from_text(self, text: str) -> str:
        """Metinden başlık çıkar (markdown **Bold** formatından)"""
        if not text:
            return ""
        
        # **Bold** formatından başlık çıkar
        bold_match = re.search(r'\*\*(.*?)\*\*', text)
        if bold_match:
            return bold_match.group(1).strip()
        
        # İlk cümleden başlık çıkar
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # İlk 50 karakteri al, çok uzunsa kes
            if len(first_sentence) > 50:
                first_sentence = first_sentence[:50].rsplit(' ', 1)[0] + "..."
            return first_sentence
        
        return "Bilinmeyen Konu"
    
    def clean_response_text(self, text: str) -> str:
        """Response metnini temizle"""
        if not text:
            return ""
        
        # Markdown gürültüsünü temizle
        cleaned = text
        
        # **Bold** işaretlerini kaldır
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        
        # *Italic* işaretlerini kaldır
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        
        # `code` işaretlerini kaldır
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
        
        # # Başlık işaretlerini kaldır
        cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
        
        # Gereksiz boşlukları temizle
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # Uzunluk sınırı uygula
        if self.max_response_chars and len(cleaned) > self.max_response_chars:
            # Son tam kelimede kes
            cleaned = cleaned[:self.max_response_chars].rsplit(' ', 1)[0] + "..."
        
        return cleaned
    
    def generate_intent(self, title: str) -> str:
        """Başlıktan intent oluştur"""
        if not title:
            return "bilinmeyen_konu"
        
        # Türkçe karakterleri değiştir
        title = title.lower()
        title = title.replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
        title = title.replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
        
        # Sadece harf ve rakamları tut
        title = re.sub(r'[^a-z0-9\s]', '', title)
        
        # Boşlukları alt çizgi ile değiştir
        title = re.sub(r'\s+', '_', title)
        
        # Çok uzunsa kısalt
        if len(title) > 30:
            words = title.split('_')
            if len(words) > 3:
                title = '_'.join(words[:3])
        
        # Prefix ekle
        if self.intent_prefix:
            title = f"{self.intent_prefix}{title}"
        
        return title
    
    def generate_prompt(self, title: str, lang: str) -> str:
        """Başlıktan prompt oluştur"""
        if not title:
            return "Bu konu hakkında bilgi verir misin?"
        
        import random
        
        if lang == "tr":
            template = random.choice(self.tr_prompts)
        else:
            template = random.choice(self.en_prompts)
        
        return template.format(title=title)
    
    def detect_language(self, text: str) -> str:
        """Metnin dilini tespit et"""
        if self.lang and self.lang != "auto":
            return self.lang
        
        try:
            # İlk 1000 karakteri kullan (daha hızlı)
            sample_text = text[:1000] if text else ""
            if sample_text:
                detected_lang = detect(sample_text)
                # Dil kodunu 2 harfe çevir
                return detected_lang[:2] if detected_lang else "en"
        except (LangDetectException, Exception):
            pass
        
        return "en"  # Varsayılan
    
    def process_row(self, row: pd.Series) -> Optional[Dict]:
        """Tek bir satırı işle"""
        try:
            # Metin verilerini al
            text = str(row.get(self.text_col, ""))
            title = str(row.get(self.title_col, "")) if self.title_col else ""
            body = str(row.get(self.body_col, "")) if self.body_col else ""
            
            # Boş satırları atla
            if not text and not title and not body:
                return None
            
            # Başlık yoksa metinden çıkar
            if not title:
                title = self.extract_title_from_text(text or body)
            
            # Response metnini oluştur
            if body:
                response = body
            elif text:
                response = text
            else:
                response = title
            
            # Response'u temizle
            response = self.clean_response_text(response)
            
            # Boş response'ları atla
            if not response or len(response.strip()) < 10:
                return None
            
            # Dil tespiti
            lang = self.detect_language(response)
            
            # Intent oluştur
            intent = self.generate_intent(title)
            
            # Prompt oluştur
            prompt = self.generate_prompt(title, lang)
            
            # JSON objesi oluştur
            result = {
                "prompt": prompt,
                "response": response,
                "intent": intent,
                "lang": lang
            }
            
            # Validation: tüm alanlar dolu olmalı
            if all(result.values()):
                return result
            
        except Exception as e:
            print(f"⚠️ Satır işleme hatası: {e}")
        
        return None
    
    def convert_to_database(self, input_path: str, db_path: str, progress_callback=None) -> int:
        """
        Parquet dosyasını SQLite veritabanına dönüştür
        
        Args:
            input_path: Giriş parquet dosya yolu
            db_path: SQLite veritabanı yolu
            progress_callback: İlerleme callback fonksiyonu
            
        Returns:
            int: Başarıyla işlenen satır sayısı
        """
        try:
            print(f"📖 Parquet dosyası okunuyor: {input_path}")
            
            # Parquet dosyasını oku
            df = pd.read_parquet(input_path)
            print(f"✅ {len(df)} satır okundu")
            
            # Kolonları kontrol et
            available_cols = df.columns.tolist()
            print(f"📊 Mevcut kolonlar: {available_cols}")
            
            # Gerekli kolonları kontrol et
            if self.text_col not in available_cols and not (self.title_col and self.body_col):
                print(f"❌ Hata: Gerekli kolon bulunamadı!")
                print(f"   text_col: {self.text_col}")
                print(f"   title_col: {self.title_col}")
                print(f"   body_col: {self.body_col}")
                return 0
            
            # Veritabanına bağlan
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Metadata kaydı ekle
            cursor.execute('''
                INSERT INTO metadata (source_file, total_rows, processed_rows, conversion_date)
                VALUES (?, ?, 0, CURRENT_TIMESTAMP)
            ''', (os.path.basename(input_path), len(df)))
            
            metadata_id = cursor.lastrowid
            
            # JSONL dosyasına yaz
            processed_count = 0
            skipped_count = 0
            
            for idx, row in df.iterrows():
                if progress_callback:
                    progress_callback(idx + 1, len(df))
                
                result = self.process_row(row)
                if result:
                    # Veritabanına ekle
                    cursor.execute('''
                        INSERT INTO conversations (prompt, response, intent, lang)
                        VALUES (?, ?, ?, ?)
                    ''', (result['prompt'], result['response'], result['intent'], result['lang']))
                    
                    processed_count += 1
                else:
                    skipped_count += 1
            
            # Metadata güncelle
            cursor.execute('''
                UPDATE metadata SET processed_rows = ? WHERE id = ?
            ''', (processed_count, metadata_id))
            
            # Commit ve kapat
            conn.commit()
            conn.close()
            
            print(f"✅ Dönüştürme tamamlandı!")
            print(f"   📝 İşlenen: {processed_count}")
            print(f"   ⏭️  Atlanan: {skipped_count}")
            print(f"   💾 Veritabanı: {db_path}")
            
            return processed_count
            
        except Exception as e:
            print(f"❌ Dönüştürme hatası: {e}")
            return 0


def main():
    """Ana fonksiyon"""
    root = tk.Tk()
    app = ParquetConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

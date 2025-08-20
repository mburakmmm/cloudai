#!/usr/bin/env python3
"""
Data Format Converter
Farklı formatlardaki dosyaları Cloud AI eğitim formatına çevirir
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import csv
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import re


class DataConverter:
    """Veri format dönüştürücü sınıfı"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cloud AI - Veri Format Dönüştürücü")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Veri değişkenleri
        self.input_file_path = ""
        self.input_data = None
        self.input_format = ""
        self.column_mapping = {}
        
        # Cloud AI eğitim formatı
        self.target_format = {
            "prompt": "string",      # Kullanıcı sorusu/mesajı
            "response": "string",    # Bot cevabı
            "intent": "string",      # Amaç/kategori (zorunlu)
            "lang": "string"         # Dil (varsayılan: "tr")
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Grid ağırlıkları
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Başlık
        title_label = ttk.Label(
            main_frame, 
            text="🌤️ Cloud AI - Veri Format Dönüştürücü",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 1. Dosya Seçimi
        ttk.Label(main_frame, text="1. Dosya Seçimi:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 10)
        )
        
        # Dosya seçim frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(
            file_frame, 
            text="📁 Dosya Seç", 
            command=self.select_file
        ).grid(row=0, column=0, padx=(0, 10))
        
        self.file_path_label = ttk.Label(file_frame, text="Dosya seçilmedi", foreground="gray")
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # 2. Format Seçimi
        ttk.Label(main_frame, text="2. Format Seçimi:", font=("Arial", 12, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(0, 10)
        )
        
        format_frame = ttk.Frame(main_frame)
        format_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.format_var = tk.StringVar(value="auto")
        ttk.Radiobutton(
            format_frame, 
            text="Otomatik Algıla", 
            variable=self.format_var, 
            value="auto"
        ).grid(row=0, column=0, padx=(0, 20))
        ttk.Radiobutton(
            format_frame, 
            text="CSV", 
            variable=self.format_var, 
            value="csv"
        ).grid(row=0, column=1, padx=(0, 20))
        ttk.Radiobutton(
            format_frame, 
            text="JSON", 
            variable=self.format_var, 
            value="json"
        ).grid(row=0, column=2)
        
        # 3. Sütun Eşleştirme
        ttk.Label(main_frame, text="3. Sütun Eşleştirme:", font=("Arial", 12, "bold")).grid(
            row=5, column=0, sticky=tk.W, pady=(0, 10)
        )
        
        # Sütun eşleştirme frame
        column_frame = ttk.Frame(main_frame)
        column_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Hedef sütunlar
        target_columns = ["prompt", "response", "intent", "lang"]
        
        for i, target_col in enumerate(target_columns):
            ttk.Label(column_frame, text=f"{target_col}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10))
            
            # Dropdown için değişken
            var = tk.StringVar()
            setattr(self, f"{target_col}_var", var)
            
            # Dropdown
            combo = ttk.Combobox(
                column_frame, 
                textvariable=var,
                state="readonly",
                width=20
            )
            combo.grid(row=i, column=1, padx=(0, 20))
            
            # ComboBox referansını da sakla
            setattr(self, f"{target_col}_combo", combo)
            
            # Varsayılan değer - sadece lang için
            if target_col == "lang":
                var.set("tr")
            # Intent için varsayılan değer verme, otomatik eşleştirme yapsın
        
        # 4. Veri Önizleme
        ttk.Label(main_frame, text="4. Veri Önizleme:", font=("Arial", 12, "bold")).grid(
            row=7, column=0, sticky=tk.W, pady=(0, 10)
        )
        
        # Treeview (tablo)
        self.tree = ttk.Treeview(main_frame, height=8, show="headings")
        self.tree.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=8, column=3, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 5. Dönüştürme ve Kaydetme
        ttk.Label(main_frame, text="5. Dönüştürme:", font=("Arial", 12, "bold")).grid(
            row=9, column=0, sticky=tk.W, pady=(0, 10)
        )
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=(0, 20))
        
        ttk.Button(
            button_frame, 
            text="🔄 Veriyi Yükle", 
            command=self.load_data
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="💾 JSON Olarak Kaydet", 
            command=self.save_as_json
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="📊 Veri İstatistikleri", 
            command=self.show_statistics
        ).pack(side=tk.LEFT)
        
        # Durum çubuğu
        self.status_var = tk.StringVar(value="Hazır")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        status_label.grid(row=11, column=0, columnspan=3, pady=(10, 0))
        
        # Grid ağırlıkları
        main_frame.rowconfigure(8, weight=1)
    
    def select_file(self):
        """Dosya seç"""
        file_types = [
            ("Tüm Dosyalar", "*.*"),
            ("CSV Dosyaları", "*.csv"),
            ("JSON Dosyaları", "*.json"),
            ("Excel Dosyaları", "*.xlsx *.xls"),
            ("Text Dosyaları", "*.txt")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Veri dosyası seç",
            filetypes=file_types
        )
        
        if file_path:
            self.input_file_path = file_path
            self.file_path_label.config(text=os.path.basename(file_path))
            self.status_var.set(f"Dosya seçildi: {os.path.basename(file_path)}")
            
            # Format'ı otomatik algıla
            if self.format_var.get() == "auto":
                if file_path.endswith('.csv'):
                    self.format_var.set("csv")
                elif file_path.endswith('.json'):
                    self.format_var.set("json")
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.format_var.set("excel")
    
    def load_data(self):
        """Veriyi yükle"""
        if not self.input_file_path:
            messagebox.showerror("Hata", "Lütfen önce bir dosya seçin!")
            return
        
        try:
            self.status_var.set("Veri yükleniyor...")
            self.root.update()
            
            # Dosya formatına göre yükle
            if self.format_var.get() == "csv":
                self.input_data = pd.read_csv(self.input_file_path)
            elif self.format_var.get() == "json":
                with open(self.input_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.input_data = pd.DataFrame(data)
                    else:
                        self.input_data = pd.DataFrame([data])
            elif self.format_var.get() == "excel":
                self.input_data = pd.read_excel(self.input_file_path)
            else:
                # Text dosyası olarak dene
                try:
                    with open(self.input_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        data = []
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Basit parsing (tab veya virgül ile ayrılmış)
                                if '\t' in line:
                                    parts = line.split('\t')
                                elif ',' in line:
                                    parts = line.split(',')
                                else:
                                    parts = [line]
                                
                                if len(parts) >= 2:
                                    data.append({
                                        'prompt': parts[0].strip(),
                                        'response': parts[1].strip(),
                                        'intent': parts[2].strip() if len(parts) > 2 else 'chat',
                                        'lang': parts[3].strip() if len(parts) > 3 else 'tr'
                                    })
                        
                        self.input_data = pd.DataFrame(data)
                except Exception as e:
                    messagebox.showerror("Hata", f"Text dosyası okunamadı: {e}")
                    return
            
            if self.input_data is not None and not self.input_data.empty:
                # Sütun listesini güncelle
                columns = [""] + list(self.input_data.columns)
                
                print(f"🔍 Debug: DataFrame sütunları: {list(self.input_data.columns)}")
                print(f"🔍 Debug: Combo values için sütunlar: {columns}")
                
                # Dropdown'ları güncelle
                for target_col in ["prompt", "response", "intent", "lang"]:
                    combo = getattr(self, f"{target_col}_combo")  # ComboBox referansını al
                    var = getattr(self, f"{target_col}_var")      # StringVar referansını al
                    print(f"🔍 Debug: {target_col} combo güncelleniyor...")
                    
                    # ComboBox'ın values özelliğini güncelle
                    combo.configure(values=columns)
                    print(f"🔍 Debug: {target_col} combo values: {combo['values']}")
                    
                    # Otomatik eşleştirme
                    if target_col == "prompt":
                        for col in columns:
                            if col and any(keyword in col.lower() for keyword in ['soru', 'question', 'prompt', 'input']):
                                var.set(col)  # StringVar'a set et
                                print(f"🔍 Debug: {target_col} otomatik eşleşti: {col}")
                                break
                    elif target_col == "response":
                        for col in columns:
                            if col and any(keyword in col.lower() for keyword in ['cevap', 'answer', 'response', 'output']):
                                var.set(col)  # StringVar'a set et
                                print(f"🔍 Debug: {target_col} otomatik eşleşti: {col}")
                                break
                    elif target_col == "intent":
                        for col in columns:
                            if col and any(keyword in col.lower() for keyword in ['intent', 'kategori', 'category', 'type']):
                                var.set(col)  # StringVar'a set et
                                print(f"🔍 Debug: {target_col} otomatik eşleşti: {col}")
                                break
                
                # Debug: Sütun listesini yazdır
                print(f"📊 CSV sütunları: {columns}")
                print(f"📊 DataFrame shape: {self.input_data.shape}")
                
                # Veri önizleme
                self.show_preview()
                
                self.status_var.set(f"Veri yüklendi: {len(self.input_data)} satır, {len(self.input_data.columns)} sütun")
                
            else:
                messagebox.showerror("Hata", "Dosya boş veya okunamadı!")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Veri yüklenirken hata: {e}")
            self.status_var.set("Hata oluştu!")
    
    def show_preview(self):
        """Veri önizleme göster"""
        if self.input_data is None or self.input_data.empty:
            return
        
        # Treeview'ı temizle
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Sütunları ayarla
        preview_data = self.input_data.head(10)  # İlk 10 satır
        
        # Sütun başlıkları
        self.tree['columns'] = list(preview_data.columns)
        for col in preview_data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # Veriyi ekle
        for idx, row in preview_data.iterrows():
            values = [str(row[col])[:50] + "..." if len(str(row[col])) > 50 else str(row[col]) for col in preview_data.columns]
            self.tree.insert("", "end", values=values)
    
    def save_as_json(self):
        """Veriyi Cloud AI formatında JSON olarak kaydet"""
        if self.input_data is None or self.input_data.empty:
            messagebox.showerror("Hata", "Önce veri yükleyin!")
            return
        
        # Sütun eşleştirmelerini al
        mapping = {}
        for target_col in ["prompt", "response", "intent", "lang"]:
            var = getattr(self, f"{target_col}_var")
            source_col = var.get()
            if source_col:
                mapping[target_col] = source_col
        
        # Zorunlu alanları kontrol et
        required_fields = ["prompt", "response", "intent"]
        missing_fields = [field for field in required_fields if field not in mapping]
        
        if missing_fields:
            messagebox.showerror("Hata", f"Zorunlu alanlar eksik: {', '.join(missing_fields)}")
            return
        
        try:
            # Veriyi dönüştür
            converted_data = []
            
            for idx, row in self.input_data.iterrows():
                item = {}
                
                # Prompt
                if mapping["prompt"] in row:
                    item["prompt"] = str(row[mapping["prompt"]]).strip()
                
                # Response
                if mapping["response"] in row:
                    item["response"] = str(row[mapping["response"]]).strip()
                
                # Intent
                if mapping["intent"] in row:
                    item["intent"] = str(row[mapping["intent"]]).strip()
                else:
                    item["intent"] = "chat"
                
                # Lang
                if "lang" in mapping and mapping["lang"] in row:
                    item["lang"] = str(row[mapping["lang"]]).strip()
                else:
                    item["lang"] = "tr"
                
                # Boş değerleri kontrol et
                if item["prompt"] and item["response"] and item["intent"]:
                    converted_data.append(item)
            
            if not converted_data:
                messagebox.showerror("Hata", "Dönüştürülecek veri bulunamadı!")
                return
            
            # Kaydetme yolu seç
            save_path = filedialog.asksaveasfilename(
                title="JSON olarak kaydet",
                defaultextension=".json",
                filetypes=[("JSON Dosyaları", "*.json")]
            )
            
            if save_path:
                # JSON olarak kaydet
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo(
                    "Başarılı", 
                    f"Veri başarıyla kaydedildi!\n\n"
                    f"Dosya: {os.path.basename(save_path)}\n"
                    f"Kayıt sayısı: {len(converted_data)}\n"
                    f"Format: Cloud AI eğitim formatı"
                )
                
                self.status_var.set(f"JSON kaydedildi: {len(converted_data)} kayıt")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatası: {e}")
            self.status_var.set("Kaydetme hatası!")
    
    def show_statistics(self):
        """Veri istatistiklerini göster"""
        if self.input_data is None or self.input_data.empty:
            messagebox.showinfo("Bilgi", "Önce veri yükleyin!")
            return
        
        # İstatistikler
        stats = {
            "Toplam Satır": len(self.input_data),
            "Toplam Sütun": len(self.input_data.columns),
            "Sütun İsimleri": ", ".join(self.input_data.columns),
            "Veri Tipleri": ", ".join([f"{col}: {dtype}" for col, dtype in self.input_data.dtypes.items()]),
            "Boş Değerler": self.input_data.isnull().sum().sum(),
            "Bellek Kullanımı": f"{self.input_data.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        # İstatistik penceresi
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Veri İstatistikleri")
        stats_window.geometry("500x400")
        
        # Text widget
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # İstatistikleri yaz
        text_widget.insert(tk.END, "📊 VERİ İSTATİSTİKLERİ\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        for key, value in stats.items():
            text_widget.insert(tk.END, f"🔹 {key}: {value}\n\n")
        
        # İlk birkaç satır
        text_widget.insert(tk.END, "📋 İLK 5 SATIR:\n")
        text_widget.insert(tk.END, "=" * 30 + "\n")
        text_widget.insert(tk.END, self.input_data.head().to_string())
        
        text_widget.config(state=tk.DISABLED)
    
    def run(self):
        """Uygulamayı başlat"""
        self.root.mainloop()


def main():
    """Ana fonksiyon"""
    app = DataConverter()
    app.run()


if __name__ == "__main__":
    main()

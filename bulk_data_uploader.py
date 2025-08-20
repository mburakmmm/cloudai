#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud AI - Toplu Veri Yükleyici
Trainer arayüzü için ayrı bir Tkinter uygulaması
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import pandas as pd
import os
import sys
from typing import List, Dict, Any, Optional
import threading

# Proje modüllerini import et
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.database_manager import DatabaseManager


class BulkDataUploader:
    """Toplu veri yükleyici Tkinter arayüzü"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌤️ Cloud AI - Toplu Veri Yükleyici")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Veri değişkenleri
        self.input_data = None
        self.selected_file_path = None
        self.file_format = "auto"
        
        # Database manager
        self.db_manager = None
        
        # UI oluştur
        self._create_ui()
        self._setup_database()
        
    def _create_ui(self):
        """Ana UI'yi oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = ttk.Label(
            main_frame,
            text="🌤️ Cloud AI - Toplu Veri Yükleyici",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Ana notebook (sekmeler)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Sekmeleri oluştur
        self._create_file_selection_tab()
        self._create_column_mapping_tab()
        self._create_preview_tab()
        self._create_upload_tab()
        
    def _create_file_selection_tab(self):
        """Dosya seçimi sekmesi"""
        frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(frame, text="📁 Dosya Seçimi")
        
        # Dosya seçim frame
        file_frame = ttk.LabelFrame(frame, text="Dosya Seçimi", padding="15")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Dosya seç butonu
        ttk.Button(
            file_frame,
            text="📁 Dosya Seç",
            command=self._select_file,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        # Seçili dosya bilgisi
        self.file_info_label = ttk.Label(
            file_frame,
            text="Dosya seçilmedi",
            foreground="gray"
        )
        self.file_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Format seçimi
        format_frame = ttk.LabelFrame(frame, text="Dosya Formatı", padding="15")
        format_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.format_var = tk.StringVar(value="auto")
        ttk.Radiobutton(
            format_frame,
            text="Otomatik Algıla",
            variable=self.format_var,
            value="auto"
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            format_frame,
            text="CSV",
            variable=self.format_var,
            value="csv"
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            format_frame,
            text="JSON",
            variable=self.format_var,
            value="json"
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            format_frame,
            text="Excel",
            variable=self.format_var,
            value="excel"
        ).pack(side=tk.LEFT)
        
        # Veri yükle butonu
        ttk.Button(
            frame,
            text="🔄 Veriyi Yükle",
            command=self._load_data,
            style="Accent.TButton"
        ).pack(pady=20)
        
        # Durum mesajı
        self.status_label = ttk.Label(
            frame,
            text="",
            font=("Arial", 10),
            foreground="blue"
        )
        self.status_label.pack()
        
    def _create_column_mapping_tab(self):
        """Sütun eşleştirme sekmesi"""
        frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(frame, text="🔗 Sütun Eşleştirme")
        
        # Sütun eşleştirme frame
        mapping_frame = ttk.LabelFrame(frame, text="Sütun Eşleştirme", padding="15")
        mapping_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Hedef sütunlar
        target_columns = [
            ("prompt", "Soru/Prompt", "Soru, question, prompt, input gibi"),
            ("response", "Cevap/Response", "Cevap, answer, response, output gibi"),
            ("intent", "Amaç/Intent", "Kategori, category, intent, type gibi"),
            ("lang", "Dil/Language", "Dil, language, lang gibi")
        ]
        
        self.column_vars = {}
        
        # ComboBox referanslarını saklamak için
        self.column_combos = {}
        
        for i, (target_col, label, hint) in enumerate(target_columns):
            row_frame = ttk.Frame(mapping_frame)
            row_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Label
            ttk.Label(
                row_frame,
                text=f"{label}:",
                font=("Arial", 11, "bold"),
                width=15
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            # Dropdown
            var = tk.StringVar()
            self.column_vars[target_col] = var
            
            combo = ttk.Combobox(
                row_frame,
                textvariable=var,
                state="readonly",
                width=25
            )
            combo.pack(side=tk.LEFT, padx=(0, 15))
            
            # ComboBox referansını sakla
            self.column_combos[target_col] = combo
            
            # Hint
            ttk.Label(
                row_frame,
                text=hint,
                font=("Arial", 9),
                foreground="gray"
            ).pack(side=tk.LEFT)
            
            # Zorunlu işareti
            if target_col in ["prompt", "response", "intent"]:
                ttk.Label(
                    row_frame,
                    text="*",
                    font=("Arial", 12, "bold"),
                    foreground="red"
                ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Otomatik eşleştirme butonu
        ttk.Button(
            frame,
            text="🔍 Otomatik Eşleştir",
            command=self._auto_map_columns
        ).pack(pady=20)
        
    def _create_preview_tab(self):
        """Veri önizleme sekmesi"""
        frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(frame, text="👁️ Veri Önizleme")
        
        # Önizleme frame
        preview_frame = ttk.LabelFrame(frame, text="Veri Önizleme (İlk 20 satır)", padding="15")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        self.tree = ttk.Treeview(preview_frame, show="headings")
        
        # Scrollbar'lar
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
    def _create_upload_tab(self):
        """Veri yükleme sekmesi"""
        frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(frame, text="📤 Veri Yükleme")
        
        # Yükleme frame
        upload_frame = ttk.LabelFrame(frame, text="Veritabanına Yükle", padding="15")
        upload_frame.pack(fill=tk.X, pady=(0, 20))
        
        # İlerleme çubuğu
        self.progress_var = tk.StringVar(value="Hazır")
        ttk.Label(
            upload_frame,
            textvariable=self.progress_var,
            font=("Arial", 11)
        ).pack(pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            upload_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(pady=(0, 20))
        
        # Yükle butonu
        self.upload_button = ttk.Button(
            frame,
            text="📤 Veritabanına Yükle",
            command=self._upload_to_database,
            style="Accent.TButton",
            state="disabled"
        )
        self.upload_button.pack(pady=20)
        
        # Sonuç mesajı
        self.result_text = scrolledtext.ScrolledText(
            frame,
            height=10,
            width=80,
            font=("Consolas", 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def _setup_database(self):
        """Veritabanı bağlantısını kur"""
        try:
            self.db_manager = DatabaseManager()
            self.db_manager.connect_local()
            print("✅ Veritabanı bağlantısı kuruldu")
        except Exception as e:
            print(f"❌ Veritabanı bağlantı hatası: {e}")
            messagebox.showerror("Hata", f"Veritabanı bağlantısı kurulamadı: {e}")
    
    def _select_file(self):
        """Dosya seç"""
        file_types = [
            ("Tüm dosyalar", "*.*"),
            ("JSON dosyaları", "*.json"),
            ("CSV dosyaları", "*.csv"),
            ("Excel dosyaları", "*.xlsx *.xls")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Veri dosyası seç",
            filetypes=file_types
        )
        
        if file_path:
            self.selected_file_path = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            self.file_info_label.config(
                text=f"📄 {file_name} ({file_size:.1f} MB)",
                foreground="green"
            )
            
            # Format'ı otomatik algıla
            if file_path.endswith('.json'):
                self.format_var.set("json")
            elif file_path.endswith('.csv'):
                self.format_var.set("csv")
            elif file_path.endswith(('.xlsx', '.xls')):
                self.format_var.set("excel")
            else:
                self.format_var.set("auto")
    
    def _load_data(self):
        """Seçili dosyayı yükle"""
        if not self.selected_file_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir dosya seçin!")
            return
        
        try:
            self.status_label.config(text="Veri yükleniyor...", foreground="blue")
            self.root.update()
            
            # Format'ı belirle
            format_type = self.format_var.get()
            if format_type == "auto":
                if self.selected_file_path.endswith('.json'):
                    format_type = "json"
                elif self.selected_file_path.endswith('.csv'):
                    format_type = "csv"
                elif self.selected_file_path.endswith(('.xlsx', '.xls')):
                    format_type = "excel"
            
            # Dosyayı oku
            if format_type == "json":
                with open(self.selected_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.input_data = pd.DataFrame(data)
                else:
                    self.input_data = pd.DataFrame([data])
                    
            elif format_type == "csv":
                self.input_data = pd.read_csv(self.selected_file_path, encoding='utf-8')
                
            elif format_type == "excel":
                self.input_data = pd.read_excel(self.selected_file_path)
            
            # Sütunları dropdown'lara ekle
            if self.input_data is not None and not self.input_data.empty:
                columns = [""] + list(self.input_data.columns)
                
                # Sütunları güncelle
                for target_col, combo in self.column_combos.items():
                    combo.configure(values=columns)
                
                # Otomatik eşleştirme
                self._auto_map_columns()
                
                # Önizleme güncelle
                self._update_preview()
                
                # Yükle butonunu aktif et
                self.upload_button.config(state="normal")
                
                self.status_label.config(
                    text=f"✅ Veri yüklendi: {len(self.input_data)} satır, {len(self.input_data.columns)} sütun",
                    foreground="green"
                )
                
                # Sütun eşleştirme sekmesine geç
                self.notebook.select(1)
                
            else:
                raise ValueError("Dosya boş veya okunamadı!")
                
        except Exception as e:
            error_msg = f"Veri yükleme hatası: {e}"
            self.status_label.config(text=error_msg, foreground="red")
            messagebox.showerror("Hata", error_msg)
            print(f"❌ {error_msg}")
    
    def _auto_map_columns(self):
        """Sütunları otomatik eşleştir"""
        if self.input_data is None or self.input_data.empty:
            return
        
        columns = list(self.input_data.columns)
        
        # Eşleştirme kuralları
        mapping_rules = {
            "prompt": ["soru", "question", "prompt", "input", "text", "query"],
            "response": ["cevap", "answer", "response", "output", "reply", "solution"],
            "intent": ["intent", "kategori", "category", "type", "class", "group"],
            "lang": ["lang", "dil", "language", "locale"]
        }
        
        for target_col, keywords in mapping_rules.items():
            var = self.column_vars[target_col]
            
            # En iyi eşleşmeyi bul
            best_match = None
            best_score = 0
            
            for col in columns:
                col_lower = col.lower()
                score = sum(1 for keyword in keywords if keyword in col_lower)
                
                if score > best_score:
                    best_score = score
                    best_match = col
            
            if best_match:
                var.set(best_match)
                print(f"🔍 {target_col} -> {best_match} (skor: {best_score})")
    
    def _update_preview(self):
        """Önizlemeyi güncelle"""
        if self.input_data is None or self.input_data.empty:
            return
        
        # Treeview'ı temizle
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Sütunları ayarla
        preview_data = self.input_data.head(20)  # İlk 20 satır
        
        self.tree['columns'] = list(preview_data.columns)
        for col in preview_data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, minwidth=100)
        
        # Veriyi ekle
        for idx, row in preview_data.iterrows():
            values = [str(row[col])[:50] + "..." if len(str(row[col])) > 50 else str(row[col]) 
                     for col in preview_data.columns]
            self.tree.insert("", "end", values=values)
    
    def _upload_to_database(self):
        """Veriyi veritabanına yükle"""
        if self.input_data is None or self.input_data.empty:
            messagebox.showwarning("Uyarı", "Yüklenecek veri yok!")
            return
        
        # Sütun eşleştirmelerini kontrol et
        mapping = {}
        for target_col, var in self.column_vars.items():
            source_col = var.get()
            if source_col:
                mapping[target_col] = source_col
        
        # Zorunlu alanları kontrol et
        required_fields = ["prompt", "response", "intent"]
        missing_fields = [field for field in required_fields if field not in mapping]
        
        if missing_fields:
            messagebox.showerror("Hata", f"Zorunlu alanlar eksik: {', '.join(missing_fields)}")
            return
        
        # Yükleme işlemini başlat
        self.upload_button.config(state="disabled")
        self.progress_var.set("Yükleniyor...")
        self.progress_bar['value'] = 0
        
        # Ayrı thread'de yükle
        threading.Thread(target=self._upload_worker, args=(mapping,), daemon=True).start()
    
    def _upload_worker(self, mapping):
        """Yükleme işçisi (ayrı thread)"""
        try:
            # Her thread'de yeni veritabanı bağlantısı oluştur
            thread_db_manager = DatabaseManager()
            thread_db_manager.connect_local()
            
            total_count = len(self.input_data)
            added_count = 0
            error_count = 0
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "📤 Veri yükleme başladı...\n\n")
            
            for idx, row in self.input_data.iterrows():
                try:
                    # Veriyi hazırla
                    conversation_data = {
                        "prompt": str(row[mapping["prompt"]]),
                        "response": str(row[mapping["response"]]),
                        "intent": str(row[mapping["intent"]]),
                        "lang": str(row[mapping.get("lang", "tr")]) if mapping.get("lang") else "tr"
                    }
                    
                    # Veritabanına ekle (thread-local connection)
                    thread_db_manager.add_conversation(**conversation_data)
                    added_count += 1
                    
                    # İlerleme güncelle
                    progress = (idx + 1) / total_count * 100
                    self.root.after(0, lambda p=progress: self.progress_bar.config(value=p))
                    self.root.after(0, lambda c=added_count, t=total_count: 
                                 self.progress_var.set(f"{c}/{t} yüklendi..."))
                    
                    # Sonuç mesajı güncelle
                    if (idx + 1) % 10 == 0:  # Her 10 kayıtta bir güncelle
                        self.root.after(0, lambda: self.result_text.insert(tk.END, 
                            f"✅ {added_count} kayıt yüklendi...\n"))
                        self.root.after(0, self.result_text.see, tk.END)
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"❌ Satır {idx + 1} hatası: {e}\n"
                    self.root.after(0, lambda: self.result_text.insert(tk.END, error_msg))
                    self.root.after(0, self.result_text.see, tk.END)
            
            # Final sonuç
            final_msg = f"\n🎉 Yükleme tamamlandı!\n"
            final_msg += f"✅ Başarılı: {added_count}\n"
            final_msg += f"❌ Hatalı: {error_count}\n"
            final_msg += f"📊 Toplam: {total_count}\n"
            
            self.root.after(0, lambda: self.result_text.insert(tk.END, final_msg))
            self.root.after(0, self.result_text.see, tk.END)
            
            # UI güncelle
            self.root.after(0, lambda: self.progress_var.set(f"Tamamlandı: {added_count}/{total_count}"))
            self.root.after(0, lambda: self.upload_button.config(state="normal"))
            
            # Başarı mesajı
            if error_count == 0:
                self.root.after(0, lambda: messagebox.showinfo("Başarılı", 
                    f"Tüm veriler başarıyla yüklendi!\nToplam: {added_count} kayıt"))
            else:
                self.root.after(0, lambda: messagebox.showwarning("Kısmen Başarılı", 
                    f"Yükleme tamamlandı!\nBaşarılı: {added_count}\nHatalı: {error_count}"))
                
        except Exception as e:
            error_msg = f"Genel yükleme hatası: {e}"
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"❌ {error_msg}\n"))
            self.root.after(0, lambda: self.progress_var.set("Hata oluştu!"))
            self.root.after(0, lambda: self.upload_button.config(state="normal"))
            self.root.after(0, lambda: messagebox.showerror("Hata", error_msg))
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.root.mainloop()


if __name__ == "__main__":
    app = BulkDataUploader()
    app.run()

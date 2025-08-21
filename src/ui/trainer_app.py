import flet as ft
import asyncio
import threading
from typing import List, Dict, Optional
import json
import os
import torch
import pandas as pd
from datetime import datetime

# Import path'i ayarla
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import'ları yap
from src.data.database_manager import DatabaseManager
from src.model.tokenizer import CustomTokenizer
from src.model.transformer_model import GenerativeTransformer
from src.data.data_loader import ConversationDataset
from src.training.trainer import Trainer
from src.training.config import get_config, get_model_config, get_training_config, get_paths_config

class TrainerApp:
    """
    Flet tabanlı eğitim ve veri yönetimi arayüzü.
    """
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.config = get_config()
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        self.paths_config = get_paths_config()
        
        # Veritabanı yöneticisi
        self.db_manager = None
        
        # Eğitim bileşenleri
        self.trainer = None
        self.tokenizer = None
        self.model = None
        
        # Eğitim durumu
        self.is_training = False
        self.is_tokenizer_training = False
        
        # Arayüz bileşenlerini hemen tanımla
        self.progress_bar = ft.ProgressBar(visible=False)
        self.status_text = ft.Text("Hazır", color=ft.Colors.GREY_600)
        self.log_text = ft.TextField(
            value="Eğitim log'ları burada görünecek...",
            multiline=True,
            read_only=True,
            min_lines=10,
            max_lines=20,
            text_size=12,
            bgcolor=ft.Colors.GREY_50,
            border_color=ft.Colors.GREY_300
        )
        
        # UI referansları
        self.prompt_input = ft.Ref[ft.TextField]()
        self.response_input = ft.Ref[ft.TextField]()
        self.intent_input = ft.Ref[ft.TextField]()
        self.lang_dropdown = ft.Ref[ft.Dropdown]()
        self.result_message = ft.Ref[ft.Container]()
        self.progress_bar_ref = ft.Ref[ft.ProgressBar]()
        self.progress_text = ft.Ref[ft.Text]()
        self.log_text_ref = ft.Ref[ft.TextField]()
        self.model_train_button = ft.Ref[ft.ElevatedButton]()
        
        # İstatistik referansları
        self.total_conversations_text = ft.Ref[ft.Text]()
        self.today_conversations_text = ft.Ref[ft.Text]()
        self.week_conversations_text = ft.Ref[ft.Text]()
        
        # Arayüzü başlat
        self._initialize_ui()
        
        # Sayfa açıldığında verileri yükle
        self.page.on_view_pop = self._on_view_pop
    
    def _initialize_ui(self):
        """Arayüzü başlat."""
        # Veritabanı bağlantısını kur
        try:
            self.db_manager = DatabaseManager()
            self.db_manager.create_tables_if_not_exist()
            print("✅ Veritabanı bağlantısı kuruldu")
        except Exception as e:
            print(f"❌ Veritabanı bağlantı hatası: {e}")
    
    def build(self):
        """Ana arayüzü oluştur."""
        return ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            ft.Icons.SCHOOL,
                            color=ft.Colors.GREEN_600,
                            size=32
                        ),
                        ft.Text(
                            "Cloud AI Trainer",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREEN_700
                        )
                    ]),
                    padding=ft.padding.all(20),
                    bgcolor=ft.Colors.GREEN_50,
                    border_radius=10
                ),
                
                # Sekmeler
                ft.Tabs(
                    selected_index=0,
                    animation_duration=300,
                    tabs=[
                        # Veri Ekleme Sekmesi
                        ft.Tab(
                            text="Veri Ekleme",
                            icon=ft.Icons.ADD_CIRCLE,
                            content=ft.Column(
                                controls=[self._build_add_data_tab()],
                                scroll=ft.ScrollMode.ALWAYS,
                                expand=True
                            )
                        ),
                        
                        # Eğitim Sekmesi
                        ft.Tab(
                            text="Eğitim",
                            icon=ft.Icons.SCHOOL,
                            content=ft.Column(
                                controls=[self._build_training_tab()],
                                scroll=ft.ScrollMode.ALWAYS,
                                expand=True
                            )
                        ),
                        
                        # Veri Görüntüleme Sekmesi
                        ft.Tab(
                            text="Veri Görüntüleme",
                            icon=ft.Icons.VISIBILITY,
                            content=ft.Column(
                                controls=[self._build_view_data_tab()],
                                scroll=ft.ScrollMode.ALWAYS,
                                expand=True
                            )
                        ),
                        

                    ]
                )
            ]),
            padding=ft.padding.all(20)
        )
    
    def _build_add_data_tab(self):
        """Veri ekleme sekmesini oluştur."""
        return ft.Container(
            content=ft.Column([
                ft.Text(
                    "Yeni Konuşma Verisi Ekle",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN_700
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Prompt alanı
                ft.TextField(
                    label="Kullanıcı Mesajı (Prompt)",
                    hint_text="Kullanıcının söyleyeceği mesajı buraya yazın...",
                    multiline=True,
                    min_lines=3,
                    max_lines=5,
                    border_color=ft.Colors.GREEN_300,
                    focused_border_color=ft.Colors.GREEN_500,
                    on_change=self._on_prompt_change
                ),
                
                ft.Container(height=15),  # Boşluk
                
                # Response alanı
                ft.TextField(
                    label="Bot Cevabı (Response)",
                    hint_text="Bot'un vereceği cevabı buraya yazın...",
                    multiline=True,
                    min_lines=3,
                    max_lines=5,
                    border_color=ft.Colors.GREEN_300,
                    focused_border_color=ft.Colors.GREEN_500,
                    on_change=self._on_response_change
                ),
                
                ft.Container(height=15),  # Boşluk
                
                # Intent alanı (zorunlu)
                ft.TextField(
                    label="Amaç (Intent) *",
                    hint_text="Bu konuşmanın amacı nedir? (örn: greeting, farewell, help_request, weather_inquiry)",
                    border_color=ft.Colors.RED_300,
                    focused_border_color=ft.Colors.RED_500,
                    on_change=self._on_intent_change
                ),
                
                ft.Container(height=15),  # Boşluk
                
                # Dil alanı
                ft.Dropdown(
                    label="Dil (Language)",
                    hint_text="Bu konuşmanın dili nedir?",
                    options=[
                        ft.dropdown.Option("tr", "Türkçe"),
                        ft.dropdown.Option("en", "English"),
                        ft.dropdown.Option("de", "Deutsch"),
                        ft.dropdown.Option("fr", "Français"),
                        ft.dropdown.Option("es", "Español"),
                        ft.dropdown.Option("it", "Italiano"),
                        ft.dropdown.Option("ru", "Русский"),
                        ft.dropdown.Option("ar", "العربية"),
                        ft.dropdown.Option("zh", "中文"),
                        ft.dropdown.Option("ja", "日本語"),
                        ft.dropdown.Option("ko", "한국어")
                    ],
                    value="tr",
                    border_color=ft.Colors.GREEN_300,
                    focused_border_color=ft.Colors.GREEN_500,
                    on_change=self._on_lang_change
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Butonlar
                ft.Row([
                    ft.ElevatedButton(
                        text="Veritabanına Ekle",
                        icon=ft.Icons.SAVE,
                        bgcolor=ft.Colors.GREEN_600,
                        color=ft.Colors.WHITE,
                        on_click=self._add_conversation
                    ),
                    
                    ft.OutlinedButton(
                        text="Temizle",
                        icon=ft.Icons.CLEAR,
                        on_click=self._clear_form
                    )
                ], alignment=ft.MainAxisAlignment.CENTER)
            ]),
            padding=ft.padding.all(20)
        )
    
    def _build_training_tab(self):
        """Eğitim sekmesini oluştur."""
        return ft.Container(
            content=ft.Column([
                ft.Text(
                    "Model Eğitimi",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN_700
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Tokenizer eğitimi
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "1. Tokenizer Eğitimi",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_700
                        ),
                        
                        ft.Text(
                            "Önce veritabanındaki tüm konuşmaları kullanarak tokenizer'ı eğitin.",
                            color=ft.Colors.GREY_600
                        ),
                        
                        ft.ElevatedButton(
                            text="Tokenizer'ı Eğit",
                            icon=ft.Icons.TOKEN,
                            bgcolor=ft.Colors.BLUE_600,
                            color=ft.Colors.WHITE,
                            on_click=self._train_tokenizer
                        )
                    ]),
                    padding=ft.padding.all(15),
                    bgcolor=ft.Colors.BLUE_50,
                    border_radius=8
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Model eğitimi
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "2. Model Eğitimi",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.ORANGE_700
                        ),
                        
                        ft.Text(
                            "Tokenizer eğitildikten sonra modeli eğitebilirsiniz.",
                            color=ft.Colors.GREY_600
                        ),
                        
                        ft.ElevatedButton(
                            text="Modeli Eğit",
                            icon=ft.Icons.MODEL_TRAINING,
                            bgcolor=ft.Colors.ORANGE_600,
                            color=ft.Colors.WHITE,
                            on_click=self._train_model,
                            disabled=True,  # Başlangıçta devre dışı
                            ref=self.model_train_button
                        )
                    ]),
                    padding=ft.padding.all(15),
                    bgcolor=ft.Colors.ORANGE_50,
                    border_radius=8
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # İlerleme ve durum
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "Eğitim Durumu",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREY_700
                        ),
                        
                        self.progress_bar,
                        self.status_text,
                        
                        ft.Container(height=10),  # Boşluk
                        
                        ft.Text(
                            "Eğitim Log'ları:",
                            size=14,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREY_700
                        ),
                        
                        ft.Container(
                            content=self.log_text,
                            height=200,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=4
                        )
                    ]),
                    padding=ft.padding.all(15),
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=8
                )
            ]),
            padding=ft.padding.all(20)
        )
    
    def _build_view_data_tab(self):
        """Veri görüntüleme sekmesini oluştur."""
        return ft.Container(
            content=ft.Column([
                ft.Text(
                    "Veritabanındaki Veriler",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN_700
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # İstatistikler
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Toplam Konuşma", size=14, color=ft.Colors.GREY_600),
                                ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_600, ref=self.total_conversations_text)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=ft.padding.all(15),
                            bgcolor=ft.Colors.WHITE,
                            border_radius=8,
                            border=ft.border.all(1, ft.Colors.GREY_300)
                        ),
                        
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Bugün Eklenen", size=14, color=ft.Colors.GREY_600),
                                ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_600, ref=self.today_conversations_text)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=ft.padding.all(15),
                            bgcolor=ft.Colors.WHITE,
                            border_radius=8,
                            border=ft.border.all(1, ft.Colors.GREY_300)
                        ),
                        
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Bu Hafta", size=14, color=ft.Colors.GREY_600),
                                ft.Text("0", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.ORANGE_600, ref=self.week_conversations_text)
                            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            padding=ft.padding.all(15),
                            bgcolor=ft.Colors.WHITE,
                            border_radius=8,
                            border=ft.border.all(1, ft.Colors.GREY_300)
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
                    padding=ft.padding.all(15),
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=8
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Yenile butonu
                ft.ElevatedButton(
                    text="Verileri Yenile",
                    icon=ft.Icons.REFRESH,
                    bgcolor=ft.Colors.GREEN_600,
                    color=ft.Colors.WHITE,
                    on_click=self._refresh_data
                ),
                
                ft.Container(height=20),  # Boşluk
                
                # Veri listesi
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "Son Konuşmalar:",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREY_700
                        ),
                        
                        ft.Text(
                            "Henüz veri yok. Veri ekleyin veya yenileyin.",
                            color=ft.Colors.GREY_500,
                            italic=True
                        )
                    ]),
                    padding=ft.padding.all(15),
                    bgcolor=ft.Colors.WHITE,
                    border_radius=8,
                    border=ft.border.all(1, ft.Colors.GREY_300)
                )
            ]),
            padding=ft.padding.all(20)
        )
    
    # Event handlers
    def _on_prompt_change(self, e):
        """Prompt alanı değiştiğinde."""
        pass
    
    def _on_response_change(self, e):
        """Response alanı değiştiğinde."""
        pass
    
    def _on_intent_change(self, e):
        """Intent alanı değiştiğinde."""
        pass
    
    # Context alanı kaldırıldı, lang alanı eklendi
    
    def _on_lang_change(self, e):
        """Dil alanı değiştiğinde."""
        pass
    
    def _on_json_input_change(self, e):
        """JSON giriş alanı değiştiğinde çalışır"""
        pass  # Şimdilik boş
    
    def _add_conversation(self, e):
        """Yeni konuşma ekle."""
        # TODO: Implement conversation addition with intent validation
        # Intent zorunlu olmalı
        pass
    
    def _clear_form(self, e):
        """Formu temizle."""
        # TODO: Implement form clearing
        pass
    
    def _train_tokenizer(self, e):
        """Tokenizer'ı eğit."""
        print("🔧 Tokenizer eğitimi başlatılıyor...")
        
        if self.is_tokenizer_training:
            self._update_log("⚠️ Tokenizer zaten eğitiliyor!")
            return
        
        # Donanım bilgisi
        device_info = self._get_device_info()
        self._update_log(f"🖥️ Donanım: {device_info}")
        
        # Veritabanından veri kontrolü
        try:
            df = self.db_manager.get_all_conversations()
            if df.empty:
                self._update_log("❌ Veritabanında eğitim verisi bulunamadı!")
                self._show_status("Hata: Veri yok", ft.Colors.RED_600)
                return
                
            total_conversations = len(df)
            self._update_log(f"📊 Toplam {total_conversations} konuşma bulundu")
            
            # Eğitimi ayrı thread'de başlat
            def train_tokenizer_async():
                try:
                    self.is_tokenizer_training = True
                    self._show_status("Tokenizer eğitiliyor...", ft.Colors.BLUE_600)
                    self._show_progress(True)
                    
                    # Tokenizer oluştur
                    from src.model.tokenizer import CustomTokenizer
                    self.tokenizer = CustomTokenizer()
                    
                    # Veri hazırla
                    self._update_log("📝 Eğitim verisi hazırlanıyor...")
                    texts = []
                    for _, row in df.iterrows():
                        texts.append(row['prompt'])
                        texts.append(row['response'])
                    
                    self._update_log(f"✅ {len(texts)} metin hazırlandı")
                    
                    # Tokenizer'ı eğit
                    self._update_log("🔄 Tokenizer eğitimi başlıyor...")
                    vocab_size = self.model_config.get('vocab_size', 30000)
                    
                    # Tokenizer path'i önce tanımla
                    tokenizer_path = self.paths_config.get('tokenizer_path', 'models/tokenizer')
                    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                    
                    # Gerçek tokenizer eğitimi
                    self._update_log("📚 Metinler tokenize ediliyor...")
                    self.tokenizer.train(texts, save_path=tokenizer_path)
                    
                    self._update_log(f"✅ Tokenizer eğitimi tamamlandı! Vocab size: {self.tokenizer.get_vocab_size()}")
                    
                    self._update_log(f"💾 Tokenizer kaydedildi: {tokenizer_path}")
                    
                    self._update_log("✅ Tokenizer eğitimi tamamlandı!")
                    self._show_status("Tokenizer hazır ✅", ft.Colors.GREEN_600)
                    
                    # Model eğitim butonunu aktif et
                    self.model_train_button.current.disabled = False
                    self.page.update()
                    
                except Exception as ex:
                    self._update_log(f"❌ Tokenizer eğitim hatası: {ex}")
                    self._show_status("Hata!", ft.Colors.RED_600)
                finally:
                    self.is_tokenizer_training = False
                    self._show_progress(False)
                    self.page.update()
            
            # Ayrı thread'de çalıştır
            import threading
            threading.Thread(target=train_tokenizer_async, daemon=True).start()
            
        except Exception as e:
            self._update_log(f"❌ Hata: {e}")
            self._show_status("Hata!", ft.Colors.RED_600)
    
    def _train_model(self, e):
        """Modeli eğit."""
        print("🔧 Model eğitimi başlatılıyor...")
        
        if self.is_training:
            self._update_log("⚠️ Model zaten eğitiliyor!")
            return
        
        # Donanım bilgisi
        device_info = self._get_device_info()
        self._update_log(f"🖥️ Donanım: {device_info}")
        
        if not self.tokenizer:
            self._update_log("❌ Önce tokenizer'ı eğitin!")
            self._show_status("Hata: Tokenizer yok", ft.Colors.RED_600)
            return
        
        # Veritabanından veri kontrolü
        try:
            df = self.db_manager.get_all_conversations()
            if df.empty:
                self._update_log("❌ Veritabanında eğitim verisi bulunamadı!")
                self._show_status("Hata: Veri yok", ft.Colors.RED_600)
                return
                
            total_conversations = len(df)
            self._update_log(f"📊 Toplam {total_conversations} konuşma ile model eğitimi")
            
            # Eğitimi ayrı thread'de başlat
            def train_model_async():
                try:
                    self.is_training = True
                    self._show_status("Model eğitiliyor...", ft.Colors.ORANGE_600)
                    self._show_progress(True)
                    
                    # Model oluştur
                    self._update_log("🧠 Transformer modeli oluşturuluyor...")
                    from src.model.transformer_model import GenerativeTransformer
                    
                    model_params = self.model_config
                    actual_vocab_size = self.tokenizer.get_vocab_size()
                    self.model = GenerativeTransformer(
                        vocab_size=actual_vocab_size,  # Tokenizer'dan gerçek vocab size
                        d_model=model_params.get('d_model', 512),
                        nhead=model_params.get('nhead', 8),
                        num_decoder_layers=model_params.get('num_decoder_layers', 6),
                        dim_feedforward=model_params.get('dim_feedforward', 2048),
                        dropout=model_params.get('dropout', 0.1)
                    )
                    self._update_log(f"📊 Model oluşturuldu - Vocab: {actual_vocab_size}, D_model: {model_params.get('d_model', 512)}")
                    
                    # Dataset oluştur
                    self._update_log("📚 Dataset hazırlanıyor...")
                    from src.data.data_loader import ConversationDataset
                    
                    dataset = ConversationDataset(
                        dataframe=df,
                        tokenizer=self.tokenizer,
                        max_length=self.training_config.get('max_seq_length', 256)
                    )
                    
                    self._update_log(f"✅ {len(dataset)} veri hazırlandı")
                    
                    # Trainer oluştur
                    self._update_log("🏋️ Trainer başlatılıyor...")
                    from src.training.trainer import Trainer
                    
                    self.trainer = Trainer(
                        model=self.model,
                        dataset=dataset,
                        config=self.training_config
                    )
                    
                    # Gerçek model eğitimi
                    self._update_log("🔄 Model eğitimi başlıyor...")
                    
                    # Eğitim konfigürasyonu
                    training_config = {
                        'learning_rate': self.training_config.get('learning_rate', 1e-4),
                        'weight_decay': self.training_config.get('weight_decay', 0.01),
                        'num_epochs': self.training_config.get('num_epochs', 10),
                        'batch_size': self.training_config.get('batch_size', 16),
                        'max_seq_length': self.training_config.get('max_seq_length', 256)
                    }
                    
                    # Trainer ile eğitim
                    self._update_log("🏋️ Trainer başlatılıyor...")
                    results = self.trainer.train(
                        num_epochs=training_config['num_epochs'],
                        batch_size=training_config['batch_size'],
                        validation_split=0.1,
                        early_stopping_patience=5,
                        save_every=1
                    )
                    
                    self._update_log(f"✅ Eğitim tamamlandı! Final loss: {results['final_val_loss']:.4f}")
                    
                    # Model kaydet
                    model_path = self.paths_config.get('model_save_path', 'models/my_chatbot.pth')
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    self._update_log(f"💾 Model kaydediliyor: {model_path}")
                    torch.save(self.model.state_dict(), model_path)
                    
                    self._update_log("✅ Model eğitimi tamamlandı!")
                    self._show_status("Model hazır ✅", ft.Colors.GREEN_600)
                    
                except Exception as ex:
                    self._update_log(f"❌ Model eğitim hatası: {ex}")
                    self._show_status("Hata!", ft.Colors.RED_600)
                finally:
                    self.is_training = False
                    self._show_progress(False)
                    self.page.update()
            
            # Ayrı thread'de çalıştır
            import threading
            threading.Thread(target=train_model_async, daemon=True).start()
            
        except Exception as e:
            self._update_log(f"❌ Hata: {e}")
            self._show_status("Hata!", ft.Colors.RED_600)
    
    def _refresh_data(self, e):
        """Verileri yenile."""
        try:
            # Veritabanından verileri al
            df = self.db_manager.get_all_conversations()
            
            if df.empty:
                # İstatistikleri sıfırla
                self.total_conversations_text.current.value = "0"
                self.today_conversations_text.current.value = "0"
                self.week_conversations_text.current.value = "0"
                self._update_log("ℹ️ Veritabanında henüz veri yok")
            else:
                # Toplam konuşma sayısı
                total_count = len(df)
                self.total_conversations_text.current.value = str(total_count)
                
                # Bugün eklenen konuşma sayısı
                from datetime import datetime, timedelta
                today = datetime.now().date()
                today_count = len(df[pd.to_datetime(df['created_at']).dt.date == today])
                self.today_conversations_text.current.value = str(today_count)
                
                # Bu hafta eklenen konuşma sayısı
                week_ago = today - timedelta(days=7)
                week_count = len(df[pd.to_datetime(df['created_at']).dt.date >= week_ago])
                self.week_conversations_text.current.value = str(week_count)
                
                self._update_log(f"✅ Veriler yenilendi: Toplam {total_count}, Bugün {today_count}, Bu hafta {week_count}")
            
            self.page.update()
            
        except Exception as e:
            self._update_log(f"❌ Veri yenileme hatası: {e}")
            self._show_status("Hata!", ft.Colors.RED_600)
    

    

    

    

    

    

    

    

    

    

    
    def _update_log(self, message):
        """Log mesajını güncelle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        current_value = self.log_text.value
        new_message = f"[{timestamp}] {message}\n"
        self.log_text.value = current_value + new_message
        self.page.update()
        print(message)  # Console'a da yazdır
    
    def _show_status(self, status, color):
        """Durum mesajını güncelle"""
        self.status_text.value = status
        self.status_text.color = color
        self.page.update()
    
    def _show_progress(self, visible):
        """İlerleme çubuğunu göster/gizle"""
        self.progress_bar.visible = visible
        if not visible:
            self.progress_bar.value = 0
        self.page.update()
    
    def _get_device_info(self):
        """Kullanılan donanım bilgisini döndür."""
        try:
            import torch
            device_info = []
            
            # PyTorch versiyonu ve CUDA durumu
            pytorch_version = torch.__version__
            device_info.append(f"PyTorch: {pytorch_version}")
            
            # CUDA durumu
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                
                device_info.append(f"CUDA: {cuda_version}")
                device_info.append(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                device_info.append(f"GPU Count: {device_count}")
                
                # Device bilgisi ekle
                current_device = torch.cuda.current_device()
                device_info.append(f"Active Device: {current_device}")
                
                return " | ".join(device_info)
            else:
                # CUDA neden mevcut değil detaylı bilgi
                device_info.append("CUDA: Mevcut değil")
                
                # PyTorch CUDA versiyonu
                if hasattr(torch.version, 'cuda') and torch.version.cuda:
                    device_info.append(f"PyTorch CUDA: {torch.version.cuda}")
                else:
                    device_info.append("PyTorch CUDA: None (CPU versiyonu)")
                
                # CUDA toolkit kontrolü
                try:
                    import subprocess
                    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        nvcc_version = result.stdout.split('\n')[3].split('release ')[1].split(',')[0]
                        device_info.append(f"CUDA Toolkit: {nvcc_version}")
                    else:
                        device_info.append("CUDA Toolkit: Bulunamadı")
                except:
                    device_info.append("CUDA Toolkit: Kontrol edilemedi")
                
                return " | ".join(device_info)
                
        except ImportError:
            return "PyTorch yüklü değil"
    
    def _on_view_pop(self, e):
        """Sayfa açıldığında çalışır."""
        # Verileri otomatik yükle
        self._refresh_data(None)

def main():
    """Ana uygulama fonksiyonu."""
    def app(page: ft.Page):
        page.title = "Cloud AI Trainer"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 1200
        page.window_height = 900
        page.window_resizable = True
        page.padding = 0
        
        # Trainer uygulamasını başlat
        trainer = TrainerApp(page)
        page.add(trainer.build())
        
        # Sayfa güncellemelerini etkinleştir
        page.update()
    
    ft.app(target=app)

if __name__ == "__main__":
    main()

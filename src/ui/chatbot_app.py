import flet as ft
import asyncio
import threading
from typing import List, Dict, Optional
import json
import os
from datetime import datetime

try:
    from ..inference.predictor import Predictor
    from ..data.database_manager import DatabaseManager
    from ..training.config import get_config, get_paths_config
except ImportError:
    # Doğrudan çalıştırıldığında absolute import kullan
    from src.inference.predictor import Predictor
    from src.data.database_manager import DatabaseManager
    from src.training.config import get_config, get_paths_config

def create_user_message(message: str, timestamp: str = None):
    """Kullanıcı mesajı widget'ı oluştur."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    return ft.Container(
        content=ft.Column([
            ft.Text(
                message,
                color=ft.Colors.WHITE,
                size=16,
                weight=ft.FontWeight.W_500
            ),
            ft.Text(
                timestamp,
                color=ft.Colors.WHITE70,
                size=12
            )
        ], spacing=4),
        bgcolor=ft.Colors.BLUE_600,
        padding=ft.padding.all(16),
        border_radius=ft.border_radius.all(20),
        margin=ft.margin.only(left=50, right=10, top=5, bottom=5),
        alignment=ft.alignment.center_right
    )

def create_bot_message(message: str, timestamp: str = None, is_typing: bool = False):
    """Bot mesajı widget'ı oluştur."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if is_typing:
        content = ft.Row([
            ft.Text("Bot yazıyor", color=ft.Colors.GREY_600, size=16),
            ft.Container(
                content=ft.Text("...", color=ft.Colors.GREY_600, size=20)
            )
        ], spacing=8)
    else:
        content = ft.Column([
            ft.Text(
                message,
                color=ft.Colors.BLACK87,
                size=16,
                weight=ft.FontWeight.W_400
            ),
            ft.Text(
                timestamp,
                color=ft.Colors.GREY_600,
                size=12
            )
        ], spacing=4)
    
    return ft.Container(
        content=content,
        bgcolor=ft.Colors.GREY_100,
        padding=ft.padding.all(16),
        border_radius=ft.border_radius.all(20),
        margin=ft.margin.only(left=10, right=50, top=5, bottom=5),
        alignment=ft.alignment.center_left
    )

class ChatbotApp:
    """
    Flet tabanlı modern chatbot arayüzü.
    """
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.config = get_config()
        self.paths_config = get_paths_config()
        
        # Model ve veritabanı
        self.predictor = None
        self.db_manager = None
        
        # Model yükleme durumu
        self.model_loaded = False
        
        # Konuşma geçmişi
        self.conversation_history = []
        
        # Arayüz bileşenleri
        self.chat_list = ft.ListView(
            expand=True,
            spacing=10,
            auto_scroll=True
        )
        
        self.message_input = ft.TextField(
            hint_text="Mesajınızı yazın... (Enter ile gönderin)",
            expand=True,
            multiline=False,
            max_lines=3,
            text_size=16,
            border_radius=ft.border_radius.all(25),
            content_padding=ft.padding.only(left=20, right=20, top=15, bottom=15),
            on_submit=self.send_message
        )
        
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            icon_color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE_600,
            icon_size=24,
            tooltip="Gönder",
            on_click=self.send_message
        )
        
        self.load_model_button = ft.ElevatedButton(
            "🤖 Modeli Yükle",
            on_click=self.load_model,
            bgcolor=ft.Colors.GREEN_600,
            color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10)
            )
        )
        
        self.model_status = ft.Text(
            "⚠️ Model yüklenmedi",
            color=ft.Colors.ORANGE_600,
            size=14,
            weight=ft.FontWeight.W_500
        )
        
        # Hoş geldin mesajı
        welcome_message = create_bot_message(
            "Merhaba! Ben Cloud AI. Size nasıl yardımcı olabilirim?",
            "Şimdi"
        )
        self.chat_list.controls.append(welcome_message)
        
        # Ana düzeni oluştur
        self.setup_ui()
        
        # Model yükleme durumunu kontrol et
        self._check_model_status()
    
    def _check_model_status(self):
        """Model durumunu kontrol et"""
        model_path = self.paths_config.get('model_save_path', 'models/my_chatbot.pth')
        tokenizer_path = self.paths_config.get('tokenizer_path', 'models/tokenizer')
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            self.model_status.value = "✅ Model mevcut - Yüklemek için tıklayın"
            self.model_status.color = ft.Colors.GREEN_600
            self.load_model_button.disabled = False
        else:
            self.model_status.value = "❌ Model bulunamadı - Önce eğitim yapın"
            self.model_status.color = ft.Colors.RED_600
            self.load_model_button.disabled = True
        
        self.page.update()
    
    def load_model(self, e):
        """Modeli yükle"""
        try:
            self.model_status.value = "🔄 Model yükleniyor..."
            self.model_status.color = ft.Colors.BLUE_600
            self.load_model_button.disabled = True
            self.page.update()
            
            # Model yükleme işlemini ayrı thread'de yap
            def load_model_async():
                try:
                    model_path = self.paths_config.get('model_save_path', 'models/my_chatbot.pth')
                    tokenizer_path = self.paths_config.get('tokenizer_path', 'models/tokenizer')
                    
                    # Predictor oluştur
                    self.predictor = Predictor(model_path, tokenizer_path, self.config)
                    
                    # Veritabanı bağlantısı
                    self.db_manager = DatabaseManager()
                    self.db_manager.connect_local()
                    
                    self.model_loaded = True
                    
                    # UI güncelle
                    self.model_status.value = "✅ Model yüklendi!"
                    self.model_status.color = ft.Colors.GREEN_600
                    self.load_model_button.disabled = True
                    self.load_model_button.text = "🤖 Model Hazır"
                    self.load_model_button.bgcolor = ft.Colors.GREY_600
                    
                    # Hoş geldin mesajını güncelle
                    self.chat_list.controls.clear()
                    welcome_message = create_bot_message(
                        "Merhaba! Ben Cloud AI. Model yüklendi ve hazırım! Size nasıl yardımcı olabilirim?",
                        "Şimdi"
                    )
                    self.chat_list.controls.append(welcome_message)
                    
                    self.page.update()
                    
                except Exception as ex:
                    self.model_status.value = f"❌ Model yükleme hatası: {ex}"
                    self.model_status.color = ft.Colors.RED_600
                    self.load_model_button.disabled = False
                    self.page.update()
            
            # Ayrı thread'de çalıştır
            threading.Thread(target=load_model_async, daemon=True).start()
            
        except Exception as e:
            self.model_status.value = f"❌ Hata: {e}"
            self.model_status.color = ft.Colors.RED_600
            self.load_model_button.disabled = False
            self.page.update()
        
    def setup_ui(self):
        """Ana arayüzü oluştur."""
        main_container = ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            ft.Icons.CHAT_BUBBLE_OUTLINE,
                            color=ft.Colors.BLUE_600,
                            size=32
                        ),
                        ft.Text(
                            "Cloud AI Chatbot",
                            size=24,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.BLUE_600
                        ),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.padding.all(20),
                    border=ft.border.only(bottom=ft.border.BorderSide(1, ft.Colors.GREY_300))
                ),
                
                # Model yükleme alanı
                ft.Container(
                    content=ft.Row([
                        self.load_model_button,
                        self.model_status
                    ], spacing=20, alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.padding.all(20),
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=ft.border_radius.all(10),
                    margin=ft.margin.all(20)
                ),
                
                # Chat alanı
                ft.Container(
                    content=self.chat_list,
                    expand=True,
                    padding=ft.padding.all(20)
                ),
                
                # Mesaj girişi
                ft.Container(
                    content=ft.Row([
                        self.message_input,
                        self.send_button
                    ], spacing=10),
                    padding=ft.padding.all(20),
                    bgcolor=ft.Colors.WHITE,
                    border=ft.border.only(top=ft.border.BorderSide(1, ft.Colors.GREY_300))
                )
            ]),
            expand=True
        )
        
        self.page.add(main_container)
    
    async def load_model(self, e):
        """Modeli yükle."""
        try:
            self.load_model_button.disabled = True
            self.load_model_button.text = "🔄 Yükleniyor..."
            self.model_status.value = "🔄 Model yükleniyor..."
            self.model_status.color = ft.Colors.BLUE_600
            self.page.update()
            
            # Model yükleme işlemini ayrı thread'de yap
            def load_model_thread():
                try:
                    # Model yollarını kontrol et
                    model_path = self.paths_config.get('model_save_path', 'models/cloudai_model.pth')
                    tokenizer_path = self.paths_config.get('tokenizer_path', 'models/tokenizer')
                    
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
                    
                    if not os.path.exists(tokenizer_path):
                        raise FileNotFoundError(f"Tokenizer dosyası bulunamadı: {tokenizer_path}")
                    
                    # Model ve tokenizer'ı yükle
                    self.predictor = Predictor(model_path, tokenizer_path, self.config)
                    self.model_loaded = True
                    
                    # Veritabanı bağlantısını kur
                    self.db_manager = DatabaseManager()
                    self.db_manager.create_tables_if_not_exist()
                    
                except Exception as e:
                    print(f"Model yükleme hatası: {e}")
                    self.model_loaded = False
            
            # Thread'i başlat
            thread = threading.Thread(target=load_model_thread)
            thread.start()
            thread.join()
            
            if self.model_loaded:
                self.load_model_button.text = "✅ Model Yüklendi"
                self.load_model_button.bgcolor = ft.Colors.GREEN_600
                self.model_status.value = "✅ Model hazır"
                self.model_status.color = ft.Colors.GREEN_600
                
                # Hoş geldin mesajını güncelle
                self.chat_list.controls.clear()
                welcome_message = create_bot_message(
                    "Merhaba! Ben Cloud AI. Model yüklendi ve hazır. Size nasıl yardımcı olabilirim?",
                    "Şimdi"
                )
                self.chat_list.controls.append(welcome_message)
                
                print("✅ Model başarıyla yüklendi!")
            else:
                self.load_model_button.disabled = False
                self.load_model_button.text = "🤖 Modeli Yükle"
                self.model_status.value = "❌ Model yüklenemedi"
                self.model_status.color = ft.Colors.RED_600
                print("❌ Model yüklenemedi!")
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.load_model_button.disabled = False
            self.load_model_button.text = "🤖 Modeli Yükle"
            self.model_status.value = f"❌ Hata: {str(e)}"
            self.model_status.color = ft.Colors.RED_600
        
        self.page.update()
    
    async def _save_to_database(self, prompt: str, response: str):
        """Veritabanına kaydet (ana thread'de)"""
        try:
            # Her thread'de yeni database manager oluştur
            db_manager = DatabaseManager()
            db_manager.connect_local()
            db_manager.add_conversation(
                prompt=prompt,
                response=response,
                intent="chat",
                lang="tr"
            )
        except Exception as e:
            print(f"Veritabanı kayıt hatası: {e}")
    
    async def send_message(self, e):
        """Mesaj gönder."""
        message_text = self.message_input.value.strip()
        if not message_text:
            return
        
        # Mesaj girişini temizle (hem buton hem Enter için)
        self.message_input.value = ""
        
        # Kullanıcı mesajını ekle
        user_message = create_user_message(message_text)
        self.chat_list.controls.append(user_message)
        
        # Typing göstergesi ekle
        typing_message = create_bot_message("", is_typing=True)
        self.chat_list.controls.append(typing_message)
        
        self.page.update()
        
        # Bot cevabını üret
        if self.model_loaded and self.predictor:
            try:
                # Ayrı thread'de generation yap
                response = None
                def generate_response():
                    nonlocal response
                    try:
                        response = self.predictor.generate_response(message_text)
                    except Exception as e:
                        response = f"Üzgünüm, bir hata oluştu: {str(e)}"
                
                # Thread'i başlat
                thread = threading.Thread(target=generate_response)
                thread.start()
                thread.join()
                
                # Typing mesajını kaldır
                self.chat_list.controls.remove(typing_message)
                
                # Bot cevabını ekle
                bot_message = create_bot_message(response)
                self.chat_list.controls.append(bot_message)
                
                # Konuşma geçmişine ekle
                self.conversation_history.append({
                    'role': 'user',
                    'content': message_text,
                    'timestamp': datetime.now().isoformat()
                })
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Veritabanına kaydet (ana thread'de)
                if self.db_manager:
                    try:
                        # Ana thread'de veritabanı işlemi yap
                        self.page.run_task(self._save_to_database, message_text, response)
                    except Exception as e:
                        print(f"Veritabanı kayıt hatası: {e}")
                
            except Exception as e:
                # Typing mesajını kaldır
                self.chat_list.controls.remove(typing_message)
                
                # Hata mesajı ekle
                error_message = create_bot_message(f"Üzgünüm, bir hata oluştu: {str(e)}")
                self.chat_list.controls.append(error_message)
        else:
            # Typing mesajını kaldır
            self.chat_list.controls.remove(typing_message)
            
            # Model yüklenmemiş mesajı
            error_message = create_bot_message("Lütfen önce modeli yükleyin.")
            self.chat_list.controls.append(error_message)
        
        self.page.update()

def main():
    """Ana uygulama fonksiyonu."""
    def app(page: ft.Page):
        page.title = "Cloud AI Chatbot"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 800
        page.window_height = 900
        page.window_resizable = True
        page.padding = 0
        
        # Chatbot uygulamasını başlat
        chatbot = ChatbotApp(page)
        
        # Sayfa güncellemelerini etkinleştir
        page.update()
    
    ft.app(target=app)

if __name__ == "__main__":
    main()
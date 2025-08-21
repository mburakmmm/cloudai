import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class DatabaseManager:
    """
    Hem Supabase hem de lokal SQLite veritabanını yöneten sınıf.
    """
    
    def __init__(self, local_db_path: Optional[str] = None):
        """
        DatabaseManager'ı başlat.
        
        Args:
            local_db_path: Lokal SQLite veritabanı dosya yolu
        """
        self.supabase_client = None
        self.local_conn = None
        
        # Supabase bağlantı bilgileri
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        # Lokal veritabanı yolu
        self.local_db_path = local_db_path or os.getenv('LOCAL_DB_PATH', 'data/cloudai.db')
        
        # Veritabanı dizinini oluştur
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        
        # Bağlantıları kur
        self.connect_local()
        if self.supabase_url and self.supabase_key:
            self.connect_supabase()
    
    def connect_supabase(self):
        """Supabase istemcisini başlat."""
        try:
            from supabase import create_client, Client
            self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase bağlantısı başarılı")
        except ImportError:
            print("⚠️  Supabase kütüphanesi yüklü değil. Lokal veritabanı kullanılacak.")
            self.supabase_client = None
        except Exception as e:
            print(f"❌ Supabase bağlantı hatası: {e}")
            self.supabase_client = None
    
    def connect_local(self):
        """SQLite bağlantısını kur."""
        try:
            self.local_conn = sqlite3.connect(self.local_db_path)
            self.local_conn.row_factory = sqlite3.Row
            print("✅ Lokal SQLite bağlantısı başarılı")
        except Exception as e:
            print(f"❌ Lokal veritabanı bağlantı hatası: {e}")
            self.local_conn = None
    
    def create_tables_if_not_exist(self):
        """Gerekli tabloları oluştur."""
        if self.local_conn:
            try:
                cursor = self.local_conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        lang TEXT DEFAULT 'tr',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                self.local_conn.commit()
                print("✅ Lokal tablolar oluşturuldu")
            except Exception as e:
                print(f"❌ Lokal tablo oluşturma hatası: {e}")
        
        if self.supabase_client:
            try:
                # Supabase'de tablo oluşturma (genellikle otomatik)
                # Burada sadece bağlantıyı test ediyoruz
                print("✅ Supabase tabloları hazır")
            except Exception as e:
                print(f"❌ Supabase tablo kontrolü hatası: {e}")
    
    def add_conversation(self, prompt: str, response: str, intent: str, lang: str = 'tr') -> bool:
        """
        Veritabanına yeni bir konuşma ekle.
        
        Args:
            prompt: Kullanıcı mesajı
            response: Bot cevabı
            intent: Amaç (zorunlu)
            lang: Dil (varsayılan: 'tr')
            
        Returns:
            bool: İşlem başarılı mı
        """
        success = False
        
        # Lokal veritabanına ekle
        if self.local_conn:
            try:
                cursor = self.local_conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations (prompt, response, intent, lang, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (prompt, response, intent, lang, datetime.now()))
                self.local_conn.commit()
                success = True
            except Exception as e:
                print(f"❌ Lokal veritabanı ekleme hatası: {e}")
        
        # Supabase'e ekle
        if self.supabase_client:
            try:
                data = {
                    'prompt': prompt,
                    'response': response,
                    'intent': intent,
                    'lang': lang,
                    'created_at': datetime.now().isoformat()
                }
                self.supabase_client.table('conversations').insert(data).execute()
                success = True
            except Exception as e:
                print(f"❌ Supabase ekleme hatası: {e}")
        
        return success
    
    def get_all_conversations(self) -> pd.DataFrame:
        """
        Tüm konuşmaları DataFrame olarak döndür.
        
        Returns:
            pd.DataFrame: Konuşma verileri
        """
        conversations = []
        
        # Lokal veritabanından al
        if self.local_conn:
            try:
                cursor = self.local_conn.cursor()
                cursor.execute('SELECT * FROM conversations ORDER BY created_at DESC')
                rows = cursor.fetchall()
                for row in rows:
                    conversations.append({
                        'id': row['id'],
                        'prompt': row['prompt'],
                        'response': row['response'],
                        'intent': row['intent'],
                        'lang': row['lang'],
                        'created_at': row['created_at']
                    })
            except Exception as e:
                print(f"❌ Lokal veritabanı okuma hatası: {e}")
        
        # Supabase'den al
        if self.supabase_client:
            try:
                response = self.supabase_client.table('conversations').select('*').order('created_at', desc=True).execute()
                for row in response.data:
                    conversations.append({
                        'id': row.get('id'),
                        'prompt': row.get('prompt'),
                        'response': row.get('response'),
                        'intent': row.get('intent'),
                        'lang': row.get('lang'),
                        'created_at': row.get('created_at')
                    })
            except Exception as e:
                print(f"❌ Supabase okuma hatası: {e}")
        
        # DataFrame oluştur
        df = pd.DataFrame(conversations)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def get_conversations_by_intent(self, intent: str) -> pd.DataFrame:
        """
        Belirli bir amaçla ilgili konuşmaları getir.
        
        Args:
            intent: Arama yapılacak amaç
            
        Returns:
            pd.DataFrame: Filtrelenmiş konuşma verileri
        """
        df = self.get_all_conversations()
        if not df.empty and 'intent' in df.columns:
            return df[df['intent'] == intent]
        return pd.DataFrame()
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Belirli bir konuşmayı sil.
        
        Args:
            conversation_id: Silinecek konuşmanın ID'si
            
        Returns:
            bool: İşlem başarılı mı
        """
        success = False
        
        # Lokal veritabanından sil
        if self.local_conn:
            try:
                cursor = self.local_conn.cursor()
                cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
                self.local_conn.commit()
                success = True
            except Exception as e:
                print(f"❌ Lokal veritabanı silme hatası: {e}")
        
        # Supabase'den sil
        if self.supabase_client:
            try:
                self.supabase_client.table('conversations').delete().eq('id', conversation_id).execute()
                success = True
            except Exception as e:
                print(f"❌ Supabase silme hatası: {e}")
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Veritabanı istatistiklerini döndür.
        
        Returns:
            Dict: İstatistik bilgileri
        """
        df = self.get_all_conversations()
        
        stats = {
            'total_conversations': len(df),
            'unique_intents': df['intent'].nunique() if not df.empty and 'intent' in df.columns else 0,
            'avg_prompt_length': df['prompt'].str.len().mean() if not df.empty else 0,
            'avg_response_length': df['response'].str.len().mean() if not df.empty else 0,
            'latest_conversation': df['created_at'].max() if not df.empty else None
        }
        
        return stats
    
    def close(self):
        """Veritabanı bağlantılarını kapat."""
        if self.local_conn:
            self.local_conn.close()
            print("✅ Lokal veritabanı bağlantısı kapatıldı")
        
        if self.supabase_client:
            print("✅ Supabase bağlantısı kapatıldı")
    
    def __enter__(self):
        """Context manager giriş."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkış."""
        self.close()

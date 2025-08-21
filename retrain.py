"""
Continuous Training Script
Yeni eklenen verilerle modeli yeniden eğitmek (fine-tune) için script
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

# Proje root'unu path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.data.database_manager import DatabaseManager
    from src.model.tokenizer import CustomTokenizer
    from src.model.transformer_model import GenerativeTransformer
    from src.data.data_loader import ConversationDataset
    from src.training.trainer import Trainer
    from src.training.config import get_config, get_model_config, get_training_config
    from src.data.preprocessor import TextPreprocessor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Modül import hatası: {e}")
    MODULES_AVAILABLE = False


class ContinuousTrainer:
    """Sürekli eğitim sınıfı"""
    
    def __init__(self, 
                 base_model_path: str = "models/chatbot-v1",
                 fine_tune_epochs: int = 3,
                 learning_rate: float = 1e-5,
                 batch_size: int = 16):
        """
        ContinuousTrainer başlat
        
        Args:
            base_model_path: Temel model yolu
            fine_tune_epochs: Fine-tuning için epoch sayısı
            learning_rate: Fine-tuning learning rate
            batch_size: Batch boyutu
        """
        self.base_model_path = base_model_path
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Veritabanı yöneticisi
        self.db_manager = DatabaseManager()
        
        # Text preprocessor
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_html=True,
            remove_emojis=False,
            min_length=10,
            max_length=500
        )
        
        # Model ve tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Eğitim geçmişi
        self.training_history = []
        
        print(f"🚀 ContinuousTrainer başlatıldı")
        print(f"  📁 Base model: {base_model_path}")
        print(f"  🔄 Fine-tune epochs: {fine_tune_epochs}")
        print(f"  📚 Learning rate: {learning_rate}")
        print(f"  📦 Batch size: {batch_size}")
    
    def load_base_model(self) -> bool:
        """
        Temel modeli yükle
        
        Returns:
            bool: Yükleme başarılı mı
        """
        try:
            print(f"📂 Temel model yükleniyor: {self.base_model_path}")
            
            # Model konfigürasyonu
            model_config = get_model_config()
            
            # Model oluştur
            self.model = GenerativeTransformer(
                vocab_size=model_config['vocab_size'],
                d_model=model_config['d_model'],
                nhead=model_config['nhead'],
                num_decoder_layers=model_config['num_decoder_layers'],
                dim_feedforward=model_config['dim_feedforward'],
                dropout=model_config['dropout']
            )
            
            # Model ağırlıklarını yükle
            model_file = os.path.join(self.base_model_path, "pytorch_model.pth")
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
                print(f"✅ Model ağırlıkları yüklendi: {model_file}")
            else:
                print(f"⚠️ Model dosyası bulunamadı: {model_file}")
                return False
            
            # Tokenizer yükle
            tokenizer_path = os.path.join(self.base_model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = CustomTokenizer(vocab_size=model_config['vocab_size'])
                self.tokenizer.load(tokenizer_path)
                print(f"✅ Tokenizer yüklendi: {tokenizer_path}")
            else:
                print(f"⚠️ Tokenizer bulunamadı: {tokenizer_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def get_new_data_since(self, last_training_date: Optional[datetime] = None) -> List[Dict]:
        """
        Son eğitimden bu yana eklenen yeni verileri al
        
        Args:
            last_training_date: Son eğitim tarihi
            
        Returns:
            List[Dict]: Yeni veriler
        """
        try:
            # Tüm konuşmaları al
            df = self.db_manager.get_all_conversations()
            
            if df.empty:
                print("ℹ️ Veritabanında veri bulunamadı")
                return []
            
            # Tarih filtreleme
            if last_training_date:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df[df['created_at'] > last_training_date]
                print(f"📅 {last_training_date} tarihinden sonra eklenen veriler filtreleniyor...")
            
            # Veriyi dict formatına çevir
            new_data = []
            for _, row in df.iterrows():
                # Metin temizleme
                clean_prompt = self.preprocessor.clean_text(row['prompt'])
                clean_response = self.preprocessor.clean_text(row['response'])
                
                if clean_prompt and clean_response:  # Geçerli verileri filtrele
                    new_data.append({
                        'prompt': clean_prompt,
                        'response': clean_response,
                        'intent': row['intent'],
                        'lang': row.get('lang', 'tr'),
                        'created_at': row['created_at']
                    })
            
            print(f"📊 {len(new_data)} yeni veri bulundu")
            return new_data
            
        except Exception as e:
            print(f"❌ Veri alma hatası: {e}")
            return []
    
    def prepare_dataset(self, data: List[Dict]) -> ConversationDataset:
        """
        Veriyi dataset formatına hazırla
        
        Args:
            data: Ham veri
            
        Returns:
            ConversationDataset: Hazırlanmış dataset
        """
        try:
            print("📚 Dataset hazırlanıyor...")
            
            # Metinleri birleştir
            texts = []
            for item in data:
                # Prompt ve response'u birleştir
                combined_text = f"{item['prompt']} {self.tokenizer.sep_token} {item['response']}"
                texts.append(combined_text)
            
            # Dataset oluştur
            dataset = ConversationDataset(
                dataframe=None,  # DataFrame kullanmıyoruz
                texts=texts,
                tokenizer=self.tokenizer,
                max_length=256
            )
            
            print(f"✅ Dataset hazırlandı: {len(dataset)} örnek")
            return dataset
            
        except Exception as e:
            print(f"❌ Dataset hazırlama hatası: {e}")
            raise
    
    def fine_tune_model(self, dataset: ConversationDataset) -> Dict:
        """
        Modeli fine-tune et
        
        Args:
            dataset: Eğitim dataset'i
            
        Returns:
            Dict: Eğitim sonuçları
        """
        try:
            print("🔧 Model fine-tuning başlıyor...")
            
            # Eğitim konfigürasyonu
            training_config = get_training_config()
            training_config.update({
                'num_epochs': self.fine_tune_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            })
            
            # Trainer oluştur
            self.trainer = Trainer(
                model=self.model,
                dataset=dataset,
                config=training_config
            )
            
            # Fine-tuning
            results = self.trainer.train(
                num_epochs=self.fine_tune_epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                early_stopping_patience=3,
                save_every=1
            )
            
            print("✅ Fine-tuning tamamlandı!")
            return results
            
        except Exception as e:
            print(f"❌ Fine-tuning hatası: {e}")
            raise
    
    def save_fine_tuned_model(self, version: str = None) -> str:
        """
        Fine-tune edilmiş modeli kaydet
        
        Args:
            version: Model versiyonu
            
        Returns:
            str: Kayıt yolu
        """
        try:
            if version is None:
                version = f"v{len(self.training_history) + 1}"
            
            # Model kaydet
            model_dir = self.trainer.save_model(version=version, include_tokenizer=True)
            
            # Eğitim geçmişine ekle
            training_record = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'fine_tune_epochs': self.fine_tune_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'data_count': len(self.training_history),
                'model_path': model_dir
            }
            
            self.training_history.append(training_record)
            
            # Geçmişi kaydet
            self._save_training_history()
            
            print(f"💾 Fine-tuned model kaydedildi: {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"❌ Model kaydetme hatası: {e}")
            raise
    
    def _save_training_history(self):
        """Eğitim geçmişini kaydet"""
        history_file = "continuous_training_history.json"
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Eğitim geçmişi kaydedildi: {history_file}")
    
    def run_continuous_training(self) -> bool:
        """
        Sürekli eğitim döngüsünü çalıştır
        
        Returns:
            bool: Başarılı mı
        """
        try:
            print("🚀 Sürekli eğitim başlatılıyor...")
            
            # Temel modeli yükle
            if not self.load_base_model():
                print("❌ Temel model yüklenemedi!")
                return False
            
            # Son eğitim tarihini al
            last_training = None
            if self.training_history:
                last_training = datetime.fromisoformat(self.training_history[-1]['timestamp'])
                print(f"📅 Son eğitim: {last_training}")
            
            # Yeni verileri al
            new_data = self.get_new_data_since(last_training)
            
            if not new_data:
                print("ℹ️ Yeni veri bulunamadı. Eğitim gerekli değil.")
                return True
            
            # Dataset hazırla
            dataset = self.prepare_dataset(new_data)
            
            # Fine-tuning
            results = self.fine_tune_model(dataset)
            
            # Modeli kaydet
            model_path = self.save_fine_tuned_model()
            
            print(f"🎉 Sürekli eğitim tamamlandı!")
            print(f"  📁 Model: {model_path}")
            print(f"  📊 Sonuçlar: {results}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sürekli eğitim hatası: {e}")
            return False


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Cloud AI Sürekli Eğitim Script'i")
    parser.add_argument("--base-model", default="models/chatbot-v1", 
                       help="Temel model yolu")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Fine-tuning epoch sayısı")
    parser.add_argument("--lr", type=float, default=1e-5, 
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch boyutu")
    parser.add_argument("--auto", action="store_true", 
                       help="Otomatik sürekli eğitim")
    
    args = parser.parse_args()
    
    if not MODULES_AVAILABLE:
        print("❌ Gerekli modüller yüklenemedi!")
        return
    
    print("🚀 Cloud AI - Sürekli Eğitim Script'i")
    print("=" * 50)
    
    # ContinuousTrainer oluştur
    trainer = ContinuousTrainer(
        base_model_path=args.base_model,
        fine_tune_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    if args.auto:
        print("🔄 Otomatik sürekli eğitim modu...")
        while True:
            try:
                success = trainer.run_continuous_training()
                if success:
                    print("✅ Eğitim tamamlandı. 1 saat bekleniyor...")
                    time.sleep(3600)  # 1 saat bekle
                else:
                    print("❌ Eğitim başarısız. 30 dakika bekleniyor...")
                    time.sleep(1800)  # 30 dakika bekle
            except KeyboardInterrupt:
                print("\n🛑 Kullanıcı tarafından durduruldu.")
                break
            except Exception as e:
                print(f"❌ Beklenmeyen hata: {e}")
                time.sleep(1800)  # 30 dakika bekle
    else:
        # Tek seferlik eğitim
        success = trainer.run_continuous_training()
        if success:
            print("✅ Eğitim başarıyla tamamlandı!")
        else:
            print("❌ Eğitim başarısız!")
            sys.exit(1)


if __name__ == "__main__":
    main()

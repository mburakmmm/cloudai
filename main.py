#!/usr/bin/env python3
"""
Cloud AI - Ana Uygulama
PyTorch tabanlı, Transformer mimarisi kullanan, Flet GUI ile arayüzü olan 
ve Supabase/Lokal DB destekli özel bir üretken dil modeli.
"""

import sys
import os
import argparse
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Ana uygulama fonksiyonu."""
    parser = argparse.ArgumentParser(
        description="Cloud AI - Kendi AI Chatbot Projem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py                    # Ana chatbot arayüzünü başlat
  python main.py --trainer         # Eğitim arayüzünü başlat
  python main.py --chatbot         # Chatbot arayüzünü başlat
  python main.py --demo            # Demo modunda çalıştır
  python main.py --config          # Konfigürasyon bilgilerini göster
        """
    )
    
    parser.add_argument(
        '--trainer',
        action='store_true',
        help='Eğitim arayüzünü başlat'
    )
    
    parser.add_argument(
        '--chatbot',
        action='store_true',
        help='Chatbot arayüzünü başlat'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo modunda çalıştır'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Konfigürasyon bilgilerini göster'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test modunda çalıştır'
    )
    
    args = parser.parse_args()
    
    # Konfigürasyon bilgilerini göster
    if args.config:
        show_config()
        return
    
    # Test modu
    if args.test:
        run_tests()
        return
    
    # Demo modu
    if args.demo:
        run_demo()
        return
    
    # Eğitim arayüzü
    if args.trainer:
        start_trainer()
        return
    
    # Chatbot arayüzü
    if args.chatbot:
        start_chatbot()
        return
    
    # Varsayılan olarak chatbot arayüzünü başlat
    start_chatbot()

def show_config():
    """Konfigürasyon bilgilerini göster."""
    try:
        from src.training.config import get_config, get_model_config, get_training_config
        
        config = get_config()
        model_config = get_model_config()
        training_config = get_training_config()
        
        print("🚀 Cloud AI - Konfigürasyon Bilgileri")
        print("=" * 50)
        
        print("\n📊 Model Parametreleri:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        print("\n🎯 Eğitim Parametreleri:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        print("\n🔧 Sistem Bilgileri:")
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name()}")
        
        print("\n📁 Proje Yapısı:")
        project_structure = """
cloudai_g/
├── src/
│   ├── data/           # Veri yönetimi ve veritabanı
│   ├── model/          # Transformer modeli ve tokenizer
│   ├── training/       # Eğitim süreci ve konfigürasyon
│   ├── inference/      # Tahmin motoru
│   └── ui/            # Kullanıcı arayüzleri
├── main.py             # Ana uygulama
├── requirements.txt    # Python bağımlılıkları
└── README.md          # Proje dokümantasyonu
        """
        print(project_structure)
        
    except ImportError as e:
        print(f"❌ Konfigürasyon yüklenemedi: {e}")
        print("💡 Lütfen önce gerekli bağımlılıkları yükleyin: pip install -r requirements.txt")

def run_tests():
    """Test modunu çalıştır."""
    print("🧪 Cloud AI - Test Modu")
    print("=" * 30)
    
    try:
        # Temel import testleri
        print("📦 Import testleri...")
        
        from src.data.database_manager import DatabaseManager
        print("  ✅ DatabaseManager import edildi")
        
        from src.model.tokenizer import CustomTokenizer
        print("  ✅ CustomTokenizer import edildi")
        
        from src.model.transformer_model import GenerativeTransformer
        print("  ✅ GenerativeTransformer import edildi")
        
        from src.data.data_loader import ConversationDataset
        print("  ✅ ConversationDataset import edildi")
        
        from src.training.trainer import Trainer
        print("  ✅ Trainer import edildi")
        
        from src.inference.predictor import Predictor
        print("  ✅ Predictor import edildi")
        
        print("\n✅ Tüm modüller başarıyla import edildi!")
        
        # Veritabanı testi
        print("\n🗄️ Veritabanı testi...")
        try:
            db = DatabaseManager()
            db.create_tables_if_not_exist()
            print("  ✅ Veritabanı bağlantısı başarılı")
            
            # Test verisi ekle
            success = db.add_conversation(
                prompt="Merhaba, nasılsın?",
                response="Merhaba! Ben iyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
                intent="greeting",
                lang="tr"
            )
            
            if success:
                print("  ✅ Test verisi eklendi")
                
                # Veriyi oku
                df = db.get_all_conversations()
                print(f"  ✅ Veri okundu: {len(df)} kayıt")
                
                # Test verisini sil
                if not df.empty:
                    db.delete_conversation(df.iloc[0]['id'])
                    print("  ✅ Test verisi silindi")
            else:
                print("  ❌ Test verisi eklenemedi")
                
        except Exception as e:
            print(f"  ❌ Veritabanı testi başarısız: {e}")
        
        # Model testi
        print("\n🤖 Model testi...")
        try:
            model = GenerativeTransformer(
                vocab_size=1000,
                d_model=128,
                nhead=4,
                num_decoder_layers=2,
                dim_feedforward=512,
                max_seq_length=128
            )
            print("  ✅ Model oluşturuldu")
            
            param_count = model.count_parameters()
            print(f"  ✅ Model parametreleri: {param_count}")
            
        except Exception as e:
            print(f"  ❌ Model testi başarısız: {e}")
        
        print("\n🎉 Tüm testler tamamlandı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")

def run_demo():
    """Demo modunu çalıştır."""
    print("🎭 Cloud AI - Demo Modu")
    print("=" * 30)
    
    try:
        print("🚀 Demo başlatılıyor...")
        
        # Basit bir demo veri seti oluştur
        demo_data = [
            {
                "prompt": "Merhaba, nasılsın?",
                "response": "Merhaba! Ben iyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
                "intent": "greeting"
            },
            {
                "prompt": "Hava nasıl?",
                "response": "Hava durumu hakkında bilgi veremiyorum, ancak size başka konularda yardımcı olabilirim.",
                "intent": "weather"
            },
            {
                "prompt": "Teşekkür ederim",
                "response": "Rica ederim! Başka bir sorunuz varsa yardımcı olmaktan memnuniyet duyarım.",
                "intent": "gratitude"
            }
        ]
        
        print("📊 Demo veri seti:")
        for i, item in enumerate(demo_data, 1):
            print(f"  {i}. Prompt: {item['prompt']}")
            print(f"     Response: {item['response']}")
            print(f"     Intent: {item['intent']}")
            print()
        
        print("💡 Bu demo veri seti ile:")
        print("  1. Tokenizer eğitimi yapabilirsiniz")
        print("  2. Model eğitimi yapabilirsiniz")
        print("  3. Chatbot arayüzünü test edebilirsiniz")
        print()
        
        print("🎯 Demo'yu başlatmak için:")
        print("  python main.py --trainer    # Eğitim arayüzü")
        print("  python main.py --chatbot    # Chatbot arayüzü")
        
    except Exception as e:
        print(f"❌ Demo hatası: {e}")

def start_trainer():
    """Eğitim arayüzünü başlat."""
    print("🎓 Cloud AI - Eğitim Arayüzü Başlatılıyor...")
    
    try:
        from src.ui.trainer_app import main as trainer_main
        trainer_main()
    except ImportError as e:
        print(f"❌ Eğitim arayüzü başlatılamadı: {e}")
        print("💡 Lütfen gerekli bağımlılıkları yükleyin: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")

def start_chatbot():
    """Chatbot arayüzünü başlat."""
    print("💬 Cloud AI - Chatbot Arayüzü Başlatılıyor...")
    
    try:
        from src.ui.chatbot_app import main as chatbot_main
        chatbot_main()
    except ImportError as e:
        print(f"❌ Chatbot arayüzü başlatılamadı: {e}")
        print("💡 Lütfen gerekli bağımlılıkları yükleyin: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")

def show_banner():
    """Uygulama banner'ını göster."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 CLOUD AI 🚀                           ║
    ║                                                              ║
    ║        Kendi AI Chatbot Projem                              ║
    ║                                                              ║
    ║    PyTorch • Transformer • Flet • Supabase                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    show_banner()
    main()

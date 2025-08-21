"""
Chatbot Test Script
Eğitilmiş modeli test etmek için
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.inference.predictor import Predictor
    from src.training.config import get_model_config
    
    print("🧠 Chatbot Test Başlıyor...")
    
    # Konfigürasyon
    model_config = get_model_config()
    
    # Predictor oluştur
    predictor = Predictor(
        model_path="models/my_chatbot.pth",
        tokenizer_path="models/tokenizer",
        config={'model': model_config}
    )
    
    print("✅ Chatbot başarıyla yüklendi!")
    
    # Test prompt'ları
    test_prompts = [
        "Merhaba",
        "Nasılsın?",
        "Python nedir?",
        "Makine öğrenmesi hakkında bilgi ver"
    ]
    
    print("\n" + "="*50)
    print("CHATBOT TEST SONUÇLARI")
    print("="*50)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        try:
            response = predictor.generate_response(prompt, max_length=50, temperature=0.3)
            print(f"🤖 Cevap: {response}")
        except Exception as e:
            print(f"❌ Hata: {e}")
    
    print("\n🎉 Test tamamlandı!")
    
except Exception as e:
    print(f"❌ Test hatası: {e}")
    import traceback
    traceback.print_exc()

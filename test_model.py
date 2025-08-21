import torch
from src.inference.predictor import Predictor
from src.training.config import get_model_config

def test_model():
    print("🧠 Model Test Ediliyor...")
    
    # Konfigürasyon
    model_config = get_model_config()
    
    # Predictor oluştur
    predictor = Predictor(
        model_path="models/my_chatbot.pth",
        tokenizer_path="models/tokenizer",
        config={'model': model_config}
    )
    
    # Test prompt'ları
    test_prompts = [
        "Merhaba",
        "Nasılsın?",
        "Hava nasıl?",
        "Python nedir?",
        "Makine öğrenmesi hakkında bilgi ver"
    ]
    
    print("\n" + "="*50)
    print("MODEL TEST SONUÇLARI")
    print("="*50)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Farklı parametrelerle test et
        response1 = predictor.generate_response(prompt, temperature=0.3, max_length=50)
        response2 = predictor.generate_response(prompt, temperature=0.7, max_length=50)
        response3 = predictor.generate_response(prompt, temperature=1.0, max_length=50)
        
        print(f"❄️  Low Temp (0.3): {response1}")
        print(f"🌡️  Medium Temp (0.7): {response2}")
        print(f"🔥 High Temp (1.0): {response3}")
        print("-" * 40)

if __name__ == "__main__":
    test_model()

import os
from src.model.tokenizer import CustomTokenizer
from src.data.database_manager import DatabaseManager

def train_tokenizer_fixed():
    print("🔧 Sabit Vocab Size ile Tokenizer Eğitimi...")
    
    # Veritabanından veri al
    db_manager = DatabaseManager()
    df = db_manager.get_all_conversations()
    
    if df.empty:
        print("❌ Veritabanında veri bulunamadı!")
        return
    
    print(f"📊 Toplam {len(df)} konuşma bulundu")
    
    # Veri hazırla
    texts = []
    for _, row in df.iterrows():
        texts.append(row['prompt'])
        texts.append(row['response'])
    
    print(f"✅ {len(texts)} metin hazırlandı")
    
    # Tokenizer oluştur ve eğit
    vocab_size = 8000  # Sabit değer
    tokenizer = CustomTokenizer(vocab_size=vocab_size)
    
    print(f"📚 Tokenizer {vocab_size} vocab size ile eğitiliyor...")
    
    tokenizer.train(texts, save_path="models/tokenizer")
    
    print(f"✅ Tokenizer eğitimi tamamlandı! Vocab size: {tokenizer.get_vocab_size()}")
    print(f"💾 Tokenizer kaydedildi: models/tokenizer")

if __name__ == "__main__":
    train_tokenizer_fixed()

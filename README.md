# Cloud AI - Kendi AI Chatbot Projem

PyTorch tabanlı, Transformer mimarisi kullanan, Flet GUI ile arayüzü olan ve Supabase/Lokal DB destekli özel bir üretken dil modeli.

## 🚀 Özellikler

- **Transformer Mimarisi**: PyTorch ile sıfırdan geliştirilmiş
- **Özel Tokenizer**: Kendi veri setinizle eğitilebilir
- **Çift Veritabanı Desteği**: Supabase ve SQLite
- **Modern GUI**: Flet ile geliştirilmiş kullanıcı arayüzü
- **Modüler Yapı**: Kolay genişletilebilir ve özelleştirilebilir

## 📁 Proje Yapısı

```
cloudai_g/
├── src/
│   ├── data/           # Veri yönetimi ve veritabanı
│   ├── model/          # Transformer modeli ve tokenizer
│   ├── training/       # Eğitim süreci ve konfigürasyon
│   ├── inference/      # Tahmin motoru
│   └── ui/            # Kullanıcı arayüzleri
├── main.py             # Ana uygulama
├── requirements.txt    # Python bağımlılıkları
└── .env               # Ortam değişkenleri
```

## 🛠️ Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd cloudai_g
```

2. **Sanal ortam oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Ortam değişkenlerini ayarlayın:**
`.env` dosyasını oluşturun ve Supabase bilgilerinizi ekleyin:
```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
LOCAL_DB_PATH=data/cloudai.db
```

## 🚀 Kullanım

### Ana Uygulama
```bash
python main.py
```

### Eğitim Arayüzü
```bash
python -m src.ui.trainer_app
```

### Chatbot Arayüzü
```bash
python -m src.ui.chatbot_app
```

## 📚 Modüller

### MODÜL 1: Veri Yönetimi
- `DatabaseManager`: Supabase ve SQLite veritabanı yönetimi
- `ConversationDataset`: PyTorch veri yükleyici

### MODÜL 2: Model Mimarisi
- `GenerativeTransformer`: PyTorch Transformer modeli
- `CustomTokenizer`: Özel tokenizer sınıfı

### MODÜL 3: Eğitim Süreci
- `Trainer`: Model eğitim sınıfı
- `Config`: Eğitim parametreleri

### MODÜL 4: Çıkarım
- `Predictor`: Tahmin motoru

### MODÜL 5: Kullanıcı Arayüzü
- `ChatbotApp`: Ana chatbot arayüzü
- `TrainerApp`: Eğitim ve veri yönetimi arayüzü

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Proje Sahibi - [@melihburakmemis](https://github.com/melihburakmemis)

Proje Linki: [https://github.com/melihburakmemis/cloudai_g](https://github.com/melihburakmemis/cloudai_g)

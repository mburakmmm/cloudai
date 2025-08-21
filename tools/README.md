# Parquet to JSONL Converter

Bu araç, Parquet dosyalarını chatbot eğitimi için uygun JSONL formatına dönüştürür.

## 🚀 Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install pandas pyarrow langdetect
```

## 📖 Kullanım

### Temel Kullanım

```bash
python tools/convert_parquet_to_json.py --input data.parquet --output output.jsonl
```

### Gelişmiş Kullanım

```bash
python tools/convert_parquet_to_json.py \
  --input data/raw.parquet \
  --output data/converted.jsonl \
  --title-col title \
  --body-col content \
  --lang tr \
  --intent-prefix polis_ \
  --max-response-chars 1000
```

## 🔧 Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--input, -i` | Giriş parquet dosya yolu | **Gerekli** |
| `--output, -o` | Çıkış JSONL dosya yolu | **Gerekli** |
| `--text-col` | Ham metin kolonu adı | `text` |
| `--title-col` | Başlık kolonu adı | Otomatik |
| `--body-col` | İçerik kolonu adı | Otomatik |
| `--lang` | Zorla kullanılacak dil kodu | Otomatik |
| `--intent-prefix` | Intent'e eklenecek ön ek | Yok |
| `--max-response-chars` | Response uzunluk sınırı | Sınırsız |

## 📊 Giriş Formatı

Parquet dosyası şu kolonlardan en az birini içermelidir:

- `text`: Ham metin (varsayılan)
- `title` + `body`: Başlık ve içerik ayrı kolonlar
- Sadece `text`: Otomatik başlık çıkarma

## 🎯 Çıkış Formatı

Her satır şu JSON formatında olacaktır:

```json
{
  "prompt": "Polis aracı nedir ve hangi amaçlarla kullanılır?",
  "response": "Polis araçları, güvenlik ve hukuki düzenin sağlanması için kullanılan...",
  "intent": "polis_araci_bilgilendirme",
  "lang": "tr"
}
```

## 🔍 Özellikler

### Otomatik Başlık Çıkarma
- Markdown **Bold** formatından başlık çıkarır
- İlk cümleden başlık türetir
- Uzun başlıkları akıllıca kısaltır

### Akıllı Prompt Oluşturma
- Türkçe için özel prompt şablonları
- İngilizce için genel prompt şablonları
- Başlığa göre dinamik prompt üretimi

### Intent Oluşturma
- Başlıktan otomatik intent türetimi
- Türkçe karakterleri uygun hale getirme
- Snake_case formatında çıktı
- Opsiyonel prefix ekleme

### Metin Temizleme
- Markdown gürültüsünü kaldırma
- Gereksiz boşlukları temizleme
- Uzunluk sınırı uygulama
- Önemli bölümleri koruma

### Dil Tespiti
- Otomatik dil tespiti (langdetect)
- Hızlı tespit için örnekleme
- Hata durumunda varsayılan dil

## 📝 Örnekler

### Örnek 1: Basit Dönüştürme
```bash
python tools/convert_parquet_to_json.py \
  --input documents.parquet \
  --output chatbot_data.jsonl
```

### Örnek 2: Polis Verileri
```bash
python tools/convert_parquet_to_json.py \
  --input polis_verileri.parquet \
  --output polis_chatbot.jsonl \
  --title-col baslik \
  --body-col icerik \
  --lang tr \
  --intent-prefix polis_
```

### Örnek 3: İngilizce Dokümanlar
```bash
python tools/convert_parquet_to_json.py \
  --input english_docs.parquet \
  --output english_chatbot.jsonl \
  --title-col title \
  --body-col content \
  --lang en \
  --max-response-chars 800
```

## 🧪 Test

Scripti test etmek için:

```bash
# Yardım menüsü
python tools/convert_parquet_to_json.py --help

# Test dosyası ile
python tools/convert_parquet_to_json.py \
  --input test.parquet \
  --output test_output.jsonl \
  --verbose
```

## 📊 Çıktı İstatistikleri

Script çalıştığında şu bilgileri gösterir:

- 📖 Okunan satır sayısı
- 📊 Mevcut kolonlar
- 🔄 İşleme ilerlemesi
- ✅ Başarıyla işlenen satır sayısı
- ⏭️ Atlanan satır sayısı
- 💾 Çıkış dosya yolu

## ⚠️ Hata Yönetimi

- Boş satırlar otomatik atlanır
- Hatalı satırlar loglanır ve atlanır
- Gerekli kolonlar kontrol edilir
- Dosya varlığı doğrulanır

## 🔄 Sonraki Adımlar

Dönüştürülen JSONL dosyası şu şekilde kullanılabilir:

1. **Chatbot Eğitimi**: `src/data/data_loader.py` ile yükleme
2. **Veri Analizi**: Pandas ile inceleme
3. **Model Eğitimi**: Trainer UI ile eğitim
4. **API Entegrasyonu**: FastAPI ile servis

## 📞 Destek

Herhangi bir sorun yaşarsanız:

1. `--verbose` flag'i ile detaylı çıktı alın
2. Giriş dosyası formatını kontrol edin
3. Gerekli kolonların mevcut olduğundan emin olun
4. Hata mesajlarını dikkatlice okuyun

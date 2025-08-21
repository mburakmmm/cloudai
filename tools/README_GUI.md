# 🚀 Parquet to SQLite Converter GUI

Tkinter arayüzlü, parquet dosyalarını chatbot eğitimi için uygun formatta SQLite veritabanına yazan modern araç.

## ✨ Özellikler

- 🖥️ **Modern Tkinter Arayüzü**: Kullanıcı dostu, sekmeli arayüz
- 📁 **File Picker**: Kolay dosya seçimi
- 🗄️ **SQLite Entegrasyonu**: Doğrudan `2-cloudai.db` veritabanına yazma
- 👁️ **Gerçek Zamanlı Önizleme**: Dosya içeriğini önceden görme
- 📊 **İlerleme Takibi**: Dönüştürme sürecini izleme
- 🔧 **Esnek Ayarlar**: Tüm parquet.cursorrules özellikleri
- 📋 **Detaylı Sonuçlar**: İşlem sonuçlarını görüntüleme

## 🚀 Kurulum

### Gerekli Bağımlılıklar

```bash
pip install pandas pyarrow langdetect
```

**Not**: Tkinter Python'da built-in olarak gelir, ayrıca yüklemeye gerek yok.

### Çalıştırma

```bash
python tools/parquet_converter_gui.py
```

## 🖥️ Arayüz Kullanımı

### 1. Dosya Seçimi
- **"Dosya Seç"** butonuna tıklayın
- Parquet dosyasını seçin
- Dosya otomatik olarak önizlenir

### 2. Kolon Ayarları
- **Text Kolonu**: Ham metin kolonu (varsayılan: `text`)
- **Title Kolonu**: Başlık kolonu (opsiyonel)
- **Body Kolonu**: İçerik kolonu (opsiyonel)

### 3. Diğer Ayarlar
- **Dil Kodu**: `tr`, `en` veya otomatik
- **Intent Prefix**: Intent'e eklenecek ön ek
- **Max Response**: Response uzunluk sınırı

### 4. Dönüştürme
- **"🔄 Dönüştür ve Veritabanına Yaz"** butonuna tıklayın
- İlerleme çubuğunu takip edin
- Sonuçları sağ panelde görün

## 📱 Arayüz Bölümleri

### Sol Panel - Ayarlar
- 📁 Dosya seçimi
- 📊 Kolon ayarları
- 🔧 Diğer ayarlar
- 🔄 Dönüştürme butonu
- 📈 İlerleme çubuğu
- 📊 Durum bilgisi

### Sağ Panel - Sekmeler

#### 👁️ Önizleme
- Dosya bilgileri
- Kolon listesi
- İlk 5 satır önizlemesi

#### 📋 Sonuçlar
- Dönüştürme sonuçları
- Kullanılan ayarlar
- İstatistikler

#### 🗄️ Veritabanı
- Veritabanı durumu
- Tablo bilgileri
- Kayıt sayıları

## 🔧 Teknik Detaylar

### Veritabanı Yapısı

#### `conversations` Tablosu
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    intent TEXT,
    lang TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `metadata` Tablosu
```sql
CREATE TABLE metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    total_rows INTEGER,
    processed_rows INTEGER,
    conversion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Dönüştürme Süreci

1. **Dosya Okuma**: Parquet dosyası pandas ile okunur
2. **Kolon Kontrolü**: Gerekli kolonlar doğrulanır
3. **Satır İşleme**: Her satır tek tek işlenir
4. **Metin Temizleme**: Markdown gürültüsü kaldırılır
5. **Prompt Oluşturma**: Başlıktan otomatik prompt üretilir
6. **Intent Türetme**: Başlıktan intent oluşturulur
7. **Dil Tespiti**: Otomatik dil tespiti yapılır
8. **Veritabanı Yazma**: Sonuçlar SQLite'a kaydedilir

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: Polis Verileri
```
Dosya: polis_verileri.parquet
Kolonlar: title, body
Dil: tr
Intent Prefix: polis_
```

### Senaryo 2: Genel Dokümanlar
```
Dosya: documents.parquet
Kolonlar: text
Dil: auto
Intent Prefix: (boş)
```

### Senaryo 3: İngilizce İçerik
```
Dosya: english_content.parquet
Kolonlar: title, content
Dil: en
Intent Prefix: doc_
```

## ⚠️ Hata Yönetimi

- **Dosya Bulunamadı**: Dosya yolu kontrol edilir
- **Kolon Hatası**: Gerekli kolonlar doğrulanır
- **Veritabanı Hatası**: Bağlantı ve tablo kontrolü
- **İşleme Hatası**: Hatalı satırlar atlanır ve loglanır

## 🔄 Sonraki Adımlar

Dönüştürülen veriler şu şekilde kullanılabilir:

1. **Chatbot Eğitimi**: `src/data/data_loader.py` ile yükleme
2. **Veri Analizi**: SQLite sorguları ile inceleme
3. **Model Eğitimi**: Trainer UI ile eğitim
4. **API Entegrasyonu**: FastAPI ile servis

## 🛠️ Geliştirici Notları

### Threading
- Dönüştürme işlemi ayrı thread'de çalışır
- UI kilitlenmez
- İlerleme gerçek zamanlı güncellenir

### Memory Management
- Büyük dosyalar için satır satır işleme
- Garbage collection optimizasyonu
- Veritabanı bağlantısı yönetimi

### Error Handling
- Try-catch blokları
- Kullanıcı dostu hata mesajları
- Detaylı loglama

## 📞 Destek

### Yaygın Sorunlar

1. **"Dosya bulunamadı"**
   - Dosya yolunu kontrol edin
   - Dosya izinlerini kontrol edin

2. **"Kolon bulunamadı"**
   - Parquet dosyasındaki kolon adlarını kontrol edin
   - Kolon ayarlarını güncelleyin

3. **"Veritabanı hatası"**
   - `2-cloudai.db` dosyasının yazılabilir olduğundan emin olun
   - Disk alanını kontrol edin

### Debug Modu
```bash
# Detaylı loglama için
python -u tools/parquet_converter_gui.py
```

## 🎯 Gelecek Özellikler

- [ ] Batch dosya işleme
- [ ] Export seçenekleri (CSV, JSON)
- [ ] Gelişmiş filtreleme
- [ ] Veri doğrulama kuralları
- [ ] Şema önizleme
- [ ] Otomatik kolon tespiti

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

**🚀 CloudAI Projesi - Parquet Converter GUI v1.0**

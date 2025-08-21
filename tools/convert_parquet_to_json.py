#!/usr/bin/env python3
"""
Parquet to JSONL Converter
Parquet dosyalarını chatbot eğitimi için uygun JSONL formatına dönüştürür

Kullanım:
    python tools/convert_parquet_to_json.py --input data/raw.parquet --output data/converted.jsonl --title-col title --body-col body --lang tr --intent-prefix polis_
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langdetect import detect, LangDetectException


class ParquetToJsonlConverter:
    """Parquet dosyalarını JSONL formatına dönüştüren sınıf"""
    
    def __init__(self, 
                 text_col: str = "text",
                 title_col: Optional[str] = None,
                 body_col: Optional[str] = None,
                 lang: Optional[str] = None,
                 intent_prefix: Optional[str] = None,
                 max_response_chars: Optional[int] = None):
        """
        Converter'ı başlat
        
        Args:
            text_col: Ham metin kolonu adı
            title_col: Başlık kolonu adı (opsiyonel)
            body_col: İçerik kolonu adı (opsiyonel)
            lang: Zorla kullanılacak dil kodu
            intent_prefix: Intent'e eklenecek ön ek
            max_response_chars: Response uzunluk sınırı
        """
        self.text_col = text_col
        self.title_col = title_col
        self.body_col = body_col
        self.lang = lang
        self.intent_prefix = intent_prefix
        self.max_response_chars = max_response_chars
        
        # Türkçe prompt şablonları
        self.tr_prompts = [
            "{title} nedir ve hangi amaçlarla kullanılır?",
            "{title} hakkında detaylı bilgi verir misin?",
            "{title} nedir? Özellikleri ve kullanım alanları nelerdir?",
            "{title} konusunda bilgi alabilir miyim?"
        ]
        
        self.en_prompts = [
            "What is {title} and what is it used for?",
            "Can you provide detailed information about {title}?",
            "What is {title}? What are its features and applications?",
            "I would like to learn about {title}."
        ]
    
    def extract_title_from_text(self, text: str) -> str:
        """Metinden başlık çıkar (markdown **Bold** formatından)"""
        if not text:
            return ""
        
        # **Bold** formatından başlık çıkar
        bold_match = re.search(r'\*\*(.*?)\*\*', text)
        if bold_match:
            return bold_match.group(1).strip()
        
        # İlk cümleden başlık çıkar
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # İlk 50 karakteri al, çok uzunsa kes
            if len(first_sentence) > 50:
                first_sentence = first_sentence[:50].rsplit(' ', 1)[0] + "..."
            return first_sentence
        
        return "Bilinmeyen Konu"
    
    def clean_response_text(self, text: str) -> str:
        """Response metnini temizle"""
        if not text:
            return ""
        
        # Markdown gürültüsünü temizle
        cleaned = text
        
        # **Bold** işaretlerini kaldır
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        
        # *Italic* işaretlerini kaldır
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        
        # `code` işaretlerini kaldır
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
        
        # # Başlık işaretlerini kaldır
        cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
        
        # Gereksiz boşlukları temizle
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        # Uzunluk sınırı uygula
        if self.max_response_chars and len(cleaned) > self.max_response_chars:
            # Son tam kelimede kes
            cleaned = cleaned[:self.max_response_chars].rsplit(' ', 1)[0] + "..."
        
        return cleaned
    
    def generate_intent(self, title: str) -> str:
        """Başlıktan intent oluştur"""
        if not title:
            return "bilinmeyen_konu"
        
        # Türkçe karakterleri değiştir
        title = title.lower()
        title = title.replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
        title = title.replace('ö', 'o').replace('ş', 's').replace('ü', 'u')
        
        # Sadece harf ve rakamları tut
        title = re.sub(r'[^a-z0-9\s]', '', title)
        
        # Boşlukları alt çizgi ile değiştir
        title = re.sub(r'\s+', '_', title)
        
        # Çok uzunsa kısalt
        if len(title) > 30:
            words = title.split('_')
            if len(words) > 3:
                title = '_'.join(words[:3])
        
        # Prefix ekle
        if self.intent_prefix:
            title = f"{self.intent_prefix}{title}"
        
        return title
    
    def generate_prompt(self, title: str, lang: str) -> str:
        """Başlıktan prompt oluştur"""
        if not title:
            return "Bu konu hakkında bilgi verir misin?"
        
        import random
        
        if lang == "tr":
            template = random.choice(self.tr_prompts)
        else:
            template = random.choice(self.en_prompts)
        
        return template.format(title=title)
    
    def detect_language(self, text: str) -> str:
        """Metnin dilini tespit et"""
        if self.lang:
            return self.lang
        
        try:
            # İlk 1000 karakteri kullan (daha hızlı)
            sample_text = text[:1000] if text else ""
            if sample_text:
                detected_lang = detect(sample_text)
                # Dil kodunu 2 harfe çevir
                return detected_lang[:2] if detected_lang else "en"
        except (LangDetectException, Exception):
            pass
        
        return "en"  # Varsayılan
    
    def process_row(self, row: pd.Series) -> Optional[Dict]:
        """Tek bir satırı işle"""
        try:
            # Metin verilerini al
            text = str(row.get(self.text_col, ""))
            title = str(row.get(self.title_col, "")) if self.title_col else ""
            body = str(row.get(self.body_col, "")) if self.body_col else ""
            
            # Boş satırları atla
            if not text and not title and not body:
                return None
            
            # Başlık yoksa metinden çıkar
            if not title:
                title = self.extract_title_from_text(text or body)
            
            # Response metnini oluştur
            if body:
                response = body
            elif text:
                response = text
            else:
                response = title
            
            # Response'u temizle
            response = self.clean_response_text(response)
            
            # Boş response'ları atla
            if not response or len(response.strip()) < 10:
                return None
            
            # Dil tespiti
            lang = self.detect_language(response)
            
            # Intent oluştur
            intent = self.generate_intent(title)
            
            # Prompt oluştur
            prompt = self.generate_prompt(title, lang)
            
            # JSON objesi oluştur
            result = {
                "prompt": prompt,
                "response": response,
                "intent": intent,
                "lang": lang
            }
            
            # Validation: tüm alanlar dolu olmalı
            if all(result.values()):
                return result
            
        except Exception as e:
            print(f"⚠️ Satır işleme hatası: {e}")
        
        return None
    
    def convert(self, input_path: str, output_path: str) -> int:
        """
        Parquet dosyasını JSONL'e dönüştür
        
        Args:
            input_path: Giriş parquet dosya yolu
            output_path: Çıkış JSONL dosya yolu
            
        Returns:
            int: Başarıyla işlenen satır sayısı
        """
        try:
            print(f"📖 Parquet dosyası okunuyor: {input_path}")
            
            # Parquet dosyasını oku
            df = pd.read_parquet(input_path)
            print(f"✅ {len(df)} satır okundu")
            
            # Kolonları kontrol et
            available_cols = df.columns.tolist()
            print(f"📊 Mevcut kolonlar: {available_cols}")
            
            # Gerekli kolonları kontrol et
            if self.text_col not in available_cols and not (self.title_col and self.body_col):
                print(f"❌ Hata: Gerekli kolon bulunamadı!")
                print(f"   text_col: {self.text_col}")
                print(f"   title_col: {self.title_col}")
                print(f"   body_col: {self.body_col}")
                return 0
            
            # Çıkış dizinini oluştur
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # JSONL dosyasına yaz
            processed_count = 0
            skipped_count = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, row in df.iterrows():
                    if idx % 1000 == 0:
                        print(f"🔄 İşleniyor: {idx}/{len(df)}")
                    
                    result = self.process_row(row)
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        skipped_count += 1
            
            print(f"✅ Dönüştürme tamamlandı!")
            print(f"   📝 İşlenen: {processed_count}")
            print(f"   ⏭️  Atlanan: {skipped_count}")
            print(f"   💾 Çıkış: {output_path}")
            
            return processed_count
            
        except Exception as e:
            print(f"❌ Dönüştürme hatası: {e}")
            return 0


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="Parquet dosyalarını chatbot eğitimi için JSONL formatına dönüştürür",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Basit dönüştürme
  python convert_parquet_to_json.py --input data.parquet --output output.jsonl
  
  # Başlık ve içerik kolonları ile
  python convert_parquet_to_json.py --input data.parquet --output output.jsonl \\
    --title-col title --body-col content --lang tr
  
  # Intent prefix ile
  python convert_parquet_to_json.py --input data.parquet --output output.jsonl \\
    --title-col title --body-col content --lang tr --intent-prefix polis_
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Giriş parquet dosya yolu'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Çıkış JSONL dosya yolu'
    )
    
    parser.add_argument(
        '--text-col',
        default='text',
        help='Ham metin kolonu adı (varsayılan: text)'
    )
    
    parser.add_argument(
        '--title-col',
        help='Başlık kolonu adı (opsiyonel)'
    )
    
    parser.add_argument(
        '--body-col',
        help='İçerik kolonu adı (opsiyonel)'
    )
    
    parser.add_argument(
        '--lang',
        help='Zorla kullanılacak dil kodu (örn: tr, en)'
    )
    
    parser.add_argument(
        '--intent-prefix',
        help='Intent\'e eklenecek ön ek'
    )
    
    parser.add_argument(
        '--max-response-chars',
        type=int,
        help='Response uzunluk sınırı'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detaylı çıktı'
    )
    
    args = parser.parse_args()
    
    # Giriş dosyasını kontrol et
    if not Path(args.input).exists():
        print(f"❌ Hata: Giriş dosyası bulunamadı: {args.input}")
        sys.exit(1)
    
    # Converter oluştur
    converter = ParquetToJsonlConverter(
        text_col=args.text_col,
        title_col=args.title_col,
        body_col=args.body_col,
        lang=args.lang,
        intent_prefix=args.intent_prefix,
        max_response_chars=args.max_response_chars
    )
    
    # Dönüştürme işlemini başlat
    print("🚀 Parquet to JSONL Dönüştürücü Başlatılıyor...")
    print(f"📁 Giriş: {args.input}")
    print(f"📁 Çıkış: {args.output}")
    print(f"🔧 Ayarlar:")
    print(f"   text_col: {args.text_col}")
    print(f"   title_col: {args.title_col or 'Otomatik'}")
    print(f"   body_col: {args.body_col or 'Otomatik'}")
    print(f"   lang: {args.lang or 'Otomatik'}")
    print(f"   intent_prefix: {args.intent_prefix or 'Yok'}")
    print(f"   max_response_chars: {args.max_response_chars or 'Sınırsız'}")
    print("-" * 50)
    
    # Dönüştür
    processed_count = converter.convert(args.input, args.output)
    
    if processed_count > 0:
        print(f"\n🎉 Başarıyla {processed_count} satır dönüştürüldü!")
        sys.exit(0)
    else:
        print(f"\n❌ Hiçbir satır dönüştürülemedi!")
        sys.exit(1)


if __name__ == "__main__":
    main()

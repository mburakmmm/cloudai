"""
Data Preprocessing Module
Veritabanından gelen ham metni temizlemek için ön işleme fonksiyonları
"""

import re
import string
import unicodedata
from typing import List, Optional, Dict
import html

# Emoji desteği (opsiyonel)
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("⚠️ emoji kütüphanesi bulunamadı. Emoji temizleme devre dışı.")


class TextPreprocessor:
    """Metin ön işleme sınıfı"""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_html: bool = True,
                 remove_emojis: bool = False,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 normalize_unicode: bool = True,
                 remove_extra_whitespace: bool = True,
                 min_length: int = 3,
                 max_length: int = 1000):
        """
        TextPreprocessor başlat
        
        Args:
            lowercase: Metni küçük harfe çevir
            remove_html: HTML etiketlerini kaldır
            remove_emojis: Emoji'leri kaldır
            remove_numbers: Sayıları kaldır
            remove_punctuation: Noktalama işaretlerini kaldır
            normalize_unicode: Unicode normalizasyonu yap
            remove_extra_whitespace: Fazla boşlukları kaldır
            min_length: Minimum metin uzunluğu
            max_length: Maksimum metin uzunluğu
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length
        
        # Türkçe karakterler için özel regex
        self.turkish_chars = re.compile(r'[çğıöşüÇĞIİÖŞÜ]')
        
        # Sayı regex'i
        self.number_regex = re.compile(r'\d+')
        
        # Noktalama işaretleri
        self.punctuation_regex = re.compile(r'[^\w\s]')
        
        # Fazla boşluk regex'i
        self.whitespace_regex = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Metni temizle
        
        Args:
            text: Temizlenecek metin
            
        Returns:
            str: Temizlenmiş metin
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalizasyonu
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # HTML etiketlerini kaldır
        if self.normalize_unicode:
            text = html.unescape(text)
        
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        # Emoji'leri kaldır
        if self.remove_emojis and EMOJI_AVAILABLE:
            text = emoji.replace_emojis(text, replace='')
        elif self.remove_emojis and not EMOJI_AVAILABLE:
            # Basit emoji temizleme (Unicode emoji karakterleri)
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub('', text)
        
        # Küçük harfe çevir
        if self.lowercase:
            text = text.lower()
        
        # Sayıları kaldır
        if self.remove_numbers:
            text = self.number_regex.sub('', text)
        
        # Noktalama işaretlerini kaldır
        if self.remove_punctuation:
            text = self.punctuation_regex.sub('', text)
        
        # Fazla boşlukları kaldır
        if self.remove_extra_whitespace:
            text = self.whitespace_regex.sub(' ', text).strip()
        
        # Uzunluk kontrolü
        if len(text) < self.min_length:
            return ""
        
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        """HTML etiketlerini kaldır"""
        # Basit HTML tag kaldırma
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def clean_conversation(self, prompt: str, response: str) -> tuple:
        """
        Konuşma çiftini temizle
        
        Args:
            prompt: Prompt metni
            response: Cevap metni
            
        Returns:
            tuple: (temizlenmiş_prompt, temizlenmiş_response)
        """
        clean_prompt = self.clean_text(prompt)
        clean_response = self.clean_text(response)
        
        return clean_prompt, clean_response
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Metin listesini toplu olarak temizle
        
        Args:
            texts: Temizlenecek metin listesi
            
        Returns:
            List[str]: Temizlenmiş metin listesi
        """
        cleaned_texts = []
        
        for text in texts:
            cleaned = self.clean_text(text)
            if cleaned:  # Boş metinleri filtrele
                cleaned_texts.append(cleaned)
        
        return cleaned_texts
    
    def validate_text(self, text: str) -> Dict[str, bool]:
        """
        Metin kalitesini doğrula
        
        Args:
            text: Doğrulanacak metin
            
        Returns:
            Dict[str, bool]: Doğrulama sonuçları
        """
        if not text:
            return {
                'has_content': False,
                'valid_length': False,
                'has_turkish_chars': False,
                'has_numbers': False,
                'has_punctuation': False,
                'is_valid': False
            }
        
        validation = {
            'has_content': len(text.strip()) > 0,
            'valid_length': self.min_length <= len(text) <= self.max_length,
            'has_turkish_chars': bool(self.turkish_chars.search(text)),
            'has_numbers': bool(self.number_regex.search(text)),
            'has_punctuation': bool(self.punctuation_regex.search(text)),
        }
        
        # Genel geçerlilik
        validation['is_valid'] = (
            validation['has_content'] and 
            validation['valid_length']
        )
        
        return validation
    
    def get_text_statistics(self, texts: List[str]) -> Dict:
        """
        Metin listesi için istatistikler hesapla
        
        Args:
            texts: Analiz edilecek metin listesi
            
        Returns:
            Dict: İstatistikler
        """
        if not texts:
            return {}
        
        # Temizlenmiş metinler
        cleaned_texts = self.clean_batch(texts)
        
        # Uzunluk istatistikleri
        lengths = [len(text) for text in cleaned_texts]
        
        # Kelime sayısı istatistikleri
        word_counts = [len(text.split()) for text in cleaned_texts]
        
        # Türkçe karakter sayısı
        turkish_char_counts = [len(self.turkish_chars.findall(text)) for text in cleaned_texts]
        
        # Sayı sayısı
        number_counts = [len(self.number_regex.findall(text)) for text in texts]  # Orijinal metinler
        
        stats = {
            'total_texts': len(texts),
            'valid_texts': len(cleaned_texts),
            'filtered_out': len(texts) - len(cleaned_texts),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'avg_turkish_chars': sum(turkish_char_counts) / len(turkish_char_counts) if turkish_char_counts else 0,
            'avg_numbers': sum(number_counts) / len(number_counts) if number_counts else 0,
            'length_distribution': {
                'short': len([l for l in lengths if l < 50]),
                'medium': len([l for l in lengths if 50 <= l < 200]),
                'long': len([l for l in lengths if l >= 200])
            }
        }
        
        return stats


# Standalone fonksiyonlar
def clean_text(text: str, **kwargs) -> str:
    """Hızlı metin temizleme"""
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.clean_text(text)


def clean_conversation(prompt: str, response: str, **kwargs) -> tuple:
    """Hızlı konuşma temizleme"""
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.clean_conversation(prompt, response)


def validate_text(text: str, **kwargs) -> Dict[str, bool]:
    """Hızlı metin doğrulama"""
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.validate_text(text)


if __name__ == "__main__":
    # Test
    print("🧪 TextPreprocessor test ediliyor...")
    
    # Test metinleri
    test_texts = [
        "Merhaba! Nasılsın? 😊",
        "<p>Bu bir HTML metni</p>",
        "123 sayılı test",
        "Çok uzun bir metin " * 100,
        "Kısa",
        "Türkçe karakterler: çğıöşü",
        "Mixed content: Hello 123! 😎"
    ]
    
    # Preprocessor oluştur
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_html=True,
        remove_emojis=True,
        remove_numbers=False,
        remove_punctuation=False,
        min_length=5,
        max_length=500
    )
    
    print("\n📊 Orijinal metinler:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n🧹 Temizlenmiş metinler:")
    cleaned_texts = preprocessor.clean_batch(test_texts)
    for i, text in enumerate(cleaned_texts, 1):
        print(f"{i}. {text}")
    
    print("\n📈 İstatistikler:")
    stats = preprocessor.get_text_statistics(test_texts)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ TextPreprocessor test tamamlandı!")

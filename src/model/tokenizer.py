"""
Custom Tokenizer Implementation
Hugging Face transformers kütüphanesi kullanarak BPE tokenizer
"""

import os
import json
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class CustomTokenizer:
    """Özel tokenizer sınıfı - BPE tabanlı"""
    
    def __init__(self, vocab_size: int = 50000, min_frequency: int = 3):
        """
        Tokenizer başlat
        
        Args:
            vocab_size: Kelime hazinesi boyutu (17K+ veri için optimize edildi)
            min_frequency: Minimum token frekansı (daha az gürültü için)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        self.is_trained = False
        
    def train(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """Metinler üzerinden tokenizer'ı eğit"""
        print(f"🔧 Tokenizer eğitimi başlıyor... {len(texts)} metin ile")
        
        # BPE tokenizer oluştur
        tokenizer = Tokenizer(models.BPE())
        
        # Pre-tokenizer: ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Decoder: ByteLevel
        tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processor: RobertaLM
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=("</s>", 2),
            cls=("<s>", 1)
        )
        
        # Trainer - 17K+ veri için optimize edildi
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
            show_progress=True,
            limit_alphabet=1000,  # Alfabe boyutunu sınırla
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # ByteLevel alfabesi
        )
        
        # Eğitim
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # PreTrainedTokenizerFast'e çevir
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>"
        )
        
        self.is_trained = True
        print(f"✅ Tokenizer eğitimi tamamlandı! Vocab size: {self.tokenizer.vocab_size}")
        
        # Kaydet
        if save_path:
            self.save(save_path)
    
    def save(self, path: str) -> None:
        """Tokenizer'ı dosyaya kaydet"""
        if not self.is_trained:
            raise ValueError("Tokenizer henüz eğitilmemiş!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save_pretrained(path)
        print(f"💾 Tokenizer kaydedildi: {path}")
    
    def load(self, path: str) -> None:
        """Tokenizer'ı dosyadan yükle"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.is_trained = True
            print(f"📂 Tokenizer yüklendi: {path}")
        except Exception as e:
            print(f"❌ Tokenizer yükleme hatası: {e}")
            raise
    
    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        """Eğitilmiş tokenizer'ı döndür"""
        if not self.is_trained:
            raise ValueError("Tokenizer henüz eğitilmemiş!")
        return self.tokenizer
    
    @property
    def pad_token_id(self) -> int:
        """Pad token ID'sini döndür"""
        if not self.is_trained:
            return 0
        return self.tokenizer.pad_token_id or 0
    
    @property
    def eos_token_id(self) -> int:
        """EOS token ID'sini döndür"""
        if not self.is_trained:
            return 1
        return self.tokenizer.eos_token_id or 1
    
    @property
    def bos_token_id(self) -> int:
        """BOS token ID'sini döndür"""
        if not self.is_trained:
            return 1
        return self.tokenizer.bos_token_id or 1
    
    @property
    def unk_token_id(self) -> int:
        """UNK token ID'sini döndür"""
        if not self.is_trained:
            return 3
        return self.tokenizer.unk_token_id or 3
        if not self.is_trained:
            raise ValueError("Tokenizer henüz eğitilmemiş!")
        return self.tokenizer
    
    def encode(self, text: str, max_length: Optional[int] = None) -> Dict:
        """Metni tokenize et"""
        if not self.is_trained:
            raise ValueError("Tokenizer henüz eğitilmemiş!")
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
    def decode(self, token_ids: List[int]) -> str:
        """Token ID'leri metne çevir"""
        if not self.is_trained:
            raise ValueError("Tokenizer henüz eğitilmemiş!")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Vocabulary boyutunu döndür"""
        if not self.is_trained:
            return 0
        return self.tokenizer.vocab_size

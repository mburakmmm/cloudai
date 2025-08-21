"""
Conversation Dataset Implementation
PyTorch Dataset sınıfından miras alan veri yükleyici
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ConversationDataset(Dataset):
    """Konuşma verileri için PyTorch Dataset"""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 256):
        """
        ConversationDataset'i başlat
        
        Args:
            dataframe: prompt, response, intent, lang kolonları olan DataFrame
            tokenizer: Eğitilmiş tokenizer
            max_length: Maksimum token uzunluğu
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Veriyi hazırla
        self._prepare_data()
        
    def _prepare_data(self):
        """Veriyi eğitim için hazırla"""
        print(f"📚 Dataset hazırlanıyor... {len(self.dataframe)} konuşma")
        
        # Prompt ve response'ları birleştir
        self.texts = []
        for _, row in self.dataframe.iterrows():
            # Prompt ve response'u birleştir
            combined_text = f"{row['prompt']} {row['response']}"
            self.texts.append(combined_text)
        
        print(f"✅ {len(self.texts)} metin hazırlandı")
    
    def __len__(self) -> int:
        """Dataset uzunluğunu döndür"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Belirtilen index'teki veriyi döndür
        
        Args:
            idx: Veri index'i
            
        Returns:
            Dict: input_ids, attention_mask ve labels içeren tensor'lar
        """
        text = self.texts[idx]
        
        # Tokenize et
        encoding = self.tokenizer.encode(text, max_length=self.max_length)
        
        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [seq_len]
        
        # Labels oluştur (shifted input_ids)
        labels = input_ids.clone()
        
        # Causal language modeling için labels'ı kaydır
        # Son token'ı kaldır, başa padding ekle
        labels = labels[1:].clone()  # [seq_len-1]
        labels = torch.cat([labels, torch.tensor([-100])])  # -100 = ignore index
        
        # Input ve attention mask'i de max_length'e getir
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            # Input IDs'i pad et
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.get_tokenizer().pad_token_id)])
            # Attention mask'i pad et
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
        
        # Truncate eğer çok uzunsa
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        
        # Labels'ı da aynı uzunlukta tut
        if len(labels) < self.max_length:
            pad_length = self.max_length - len(labels)
            labels = torch.cat([labels, torch.full((pad_length,), -100)])
        
        labels = labels[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_sample(self, idx: int = 0) -> Dict[str, torch.Tensor]:
        """Örnek veri döndür (debug için)"""
        if idx >= len(self):
            idx = 0
        
        sample = self[idx]
        
        # Decode edilmiş metinleri de ekle
        sample['text'] = self.texts[idx]
        sample['decoded_input'] = self.tokenizer.decode(sample['input_ids'])
        sample['decoded_labels'] = self.tokenizer.decode(
            sample['labels'][sample['labels'] != -100]
        )
        
        return sample
    
    def get_statistics(self) -> Dict:
        """Dataset istatistiklerini döndür"""
        # Metin uzunlukları
        text_lengths = [len(text.split()) for text in self.texts]
        
        # Token uzunlukları
        token_lengths = []
        for text in self.texts[:100]:  # İlk 100 metin için
            encoding = self.tokenizer.encode(text)
            token_lengths.append(encoding['input_ids'].size(1))
        
        stats = {
            'total_conversations': len(self.dataframe),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'max_text_length': max(text_lengths),
            'min_text_length': min(text_lengths),
            'avg_token_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'min_token_length': min(token_lengths) if token_lengths else 0,
            'max_sequence_length': self.max_length
        }
        
        return stats
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Batch'i birleştir
        
        Args:
            batch: Dataset'ten gelen veri listesi
            
        Returns:
            Dict: Batch halinde tensor'lar
        """
        # Batch boyutunu al
        batch_size = len(batch)
        
        # Her kolonu ayrı ayrı topla
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,  # [batch_size, seq_len]
            'attention_mask': attention_mask,  # [batch_size, seq_len]
            'labels': labels  # [batch_size, seq_len]
        }


class ConversationDataLoader:
    """ConversationDataset için DataLoader wrapper"""
    
    def __init__(self, dataset: ConversationDataset, batch_size: int = 16, 
                 shuffle: bool = True, num_workers: int = 0):
        """
        DataLoader'ı başlat
        
        Args:
            dataset: ConversationDataset
            batch_size: Batch boyutu
            shuffle: Karıştır
            num_workers: Worker sayısı
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # PyTorch DataLoader oluştur
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=True
        )
    
    def __iter__(self):
        """Iterator döndür"""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Batch sayısını döndür"""
        return len(self.dataloader)
    
    def get_batch_info(self) -> Dict:
        """Batch bilgilerini döndür"""
        return {
            'batch_size': self.batch_size,
            'num_batches': len(self.dataloader),
            'total_samples': len(self.dataset),
            'shuffle': self.shuffle,
            'num_workers': self.num_workers
        }

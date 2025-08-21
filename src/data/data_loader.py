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
        
        # Sadece prompt'ları al, response'ları ayrı tut
        self.prompts = []
        self.responses = []
        for _, row in self.dataframe.iterrows():
            # Prompt ve response'u ayrı ayrı sakla
            self.prompts.append(row['prompt'])
            self.responses.append(row['response'])
        
        print(f"✅ {len(self.prompts)} prompt ve {len(self.responses)} response hazırlandı")
    
    def __len__(self) -> int:
        """Dataset uzunluğunu döndür"""
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Belirtilen index'teki veriyi döndür
        
        Args:
            idx: Veri index'i
            
        Returns:
            Dict: input_ids, attention_mask ve labels içeren tensor'lar
        """
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Prompt'u tokenize et
        prompt_encoding = self.tokenizer.encode(prompt, max_length=self.max_length//2)
        prompt_ids = prompt_encoding['input_ids'].squeeze(0)
        
        # Response'u tokenize et
        response_encoding = self.tokenizer.encode(response, max_length=self.max_length//2)
        response_ids = response_encoding['input_ids'].squeeze(0)
        
        # Prompt ve response'u birleştir (eğitim için)
        input_ids = torch.cat([prompt_ids, response_ids])
        
        # Attention mask oluştur
        attention_mask = torch.ones(len(input_ids))
        
        # Labels oluştur (sadece response kısmı için)
        labels = torch.full((len(input_ids),), -100)  # -100 = ignore index
        labels[len(prompt_ids):] = response_ids  # Sadece response kısmını label olarak kullan
        
        # Padding ve truncation
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.get_tokenizer().pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
            labels = torch.cat([labels, torch.full((pad_length,), -100)])
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
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
        sample['prompt'] = self.prompts[idx]
        sample['response'] = self.responses[idx]
        sample['decoded_input'] = self.tokenizer.decode(sample['input_ids'])
        sample['decoded_labels'] = self.tokenizer.decode(
            sample['labels'][sample['labels'] != -100]
        )
        
        return sample
    
    def get_statistics(self) -> Dict:
        """Dataset istatistiklerini döndür"""
        # Prompt ve response uzunlukları
        prompt_lengths = [len(prompt.split()) for prompt in self.prompts]
        response_lengths = [len(response.split()) for prompt in self.responses]
        
        # Token uzunlukları
        token_lengths = []
        for prompt, response in zip(self.prompts[:100], self.responses[:100]):  # İlk 100 için
            prompt_encoding = self.tokenizer.encode(prompt)
            response_encoding = self.tokenizer.encode(response)
            token_lengths.append(prompt_encoding['input_ids'].size(1) + response_encoding['input_ids'].size(1))
        
        stats = {
            'total_conversations': len(self.dataframe),
            'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths),
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'max_prompt_length': max(prompt_lengths),
            'max_response_length': max(response_lengths),
            'min_prompt_length': min(prompt_lengths),
            'min_response_length': min(response_lengths),
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

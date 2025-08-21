"""
Model Evaluation and Metrics
Model performansını ölçmek için metrik hesaplama fonksiyonları
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK bulunamadı. BLEU skoru hesaplanamayacak.")

try:
    from evaluate import load
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("⚠️ evaluate kütüphanesi bulunamadı. BLEU skoru hesaplanamayacak.")


class ModelEvaluator:
    """Model performansını değerlendiren sınıf"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Smoothing function for BLEU
        if NLTK_AVAILABLE:
            self.smoothing = SmoothingFunction()
        
        # BLEU metric from evaluate library
        if EVALUATE_AVAILABLE:
            try:
                self.bleu_metric = load("bleu")
            except:
                self.bleu_metric = None
    
    def calculate_perplexity(self, dataloader) -> float:
        """
        Modelin perplexity skorunu hesapla
        
        Args:
            dataloader: Test veri seti
            
        Returns:
            float: Perplexity skoru (düşük olması daha iyi)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Loss hesapla (CrossEntropyLoss ile aynı mantık)
                logits = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                
                # Sadece geçerli label'ları hesapla (-100 olmayan)
                valid_mask = labels_flat != -100
                if valid_mask.sum() > 0:
                    valid_logits = logits[valid_mask]
                    valid_labels = labels_flat[valid_mask]
                    
                    loss = F.cross_entropy(valid_logits, valid_labels, reduction='sum')
                    total_loss += loss.item()
                    total_tokens += valid_mask.sum().item()
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def calculate_bleu(self, generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
        """
        BLEU skorunu hesapla
        
        Args:
            generated_texts: Model tarafından üretilen metinler
            reference_texts: Referans (gerçek) metinler
            
        Returns:
            Dict: BLEU skorları
        """
        results = {}
        
        # NLTK ile BLEU hesaplama
        if NLTK_AVAILABLE:
            try:
                bleu_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    # Tokenize (basit whitespace split)
                    gen_tokens = gen.split()
                    ref_tokens = ref.split()
                    
                    # BLEU hesapla
                    score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing.method1)
                    bleu_scores.append(score)
                
                results['bleu_nltk'] = np.mean(bleu_scores)
                results['bleu_nltk_std'] = np.std(bleu_scores)
            except Exception as e:
                print(f"⚠️ NLTK BLEU hesaplama hatası: {e}")
                results['bleu_nltk'] = 0.0
        
        # Evaluate library ile BLEU hesaplama
        if EVALUATE_AVAILABLE and self.bleu_metric:
            try:
                # BLEU için referans metinleri list of list formatında olmalı
                references = [[ref] for ref in reference_texts]
                
                bleu_result = self.bleu_metric.compute(
                    predictions=generated_texts,
                    references=references
                )
                
                results['bleu_evaluate'] = bleu_result['bleu']
            except Exception as e:
                print(f"⚠️ Evaluate BLEU hesaplama hatası: {e}")
                results['bleu_evaluate'] = 0.0
        
        return results
    
    def calculate_accuracy(self, dataloader) -> float:
        """
        Token-level accuracy hesapla
        
        Args:
            dataloader: Test veri seti
            
        Returns:
            float: Accuracy skoru
        """
        self.model.eval()
        correct_tokens = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Predictions
                predictions = torch.argmax(outputs, dim=-1)
                
                # Sadece geçerli label'ları hesapla
                valid_mask = labels != -100
                if valid_mask.sum() > 0:
                    valid_preds = predictions[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    correct_tokens += (valid_preds == valid_labels).sum().item()
                    total_tokens += valid_mask.sum().item()
        
        if total_tokens == 0:
            return 0.0
        
        accuracy = correct_tokens / total_tokens
        return accuracy
    
    def generate_sample_responses(self, prompts: List[str], max_length: int = 50) -> List[str]:
        """
        Test prompt'ları için örnek cevaplar üret
        
        Args:
            prompts: Test prompt'ları
            max_length: Maksimum token sayısı
            
        Returns:
            List[str]: Üretilen cevaplar
        """
        self.model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Prompt'u tokenize et
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generation
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
        
        return responses
    
    def comprehensive_evaluation(self, dataloader, test_prompts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Kapsamlı model değerlendirmesi
        
        Args:
            dataloader: Test veri seti
            test_prompts: Test prompt'ları (BLEU için)
            
        Returns:
            Dict: Tüm metrikler
        """
        print("🔍 Model değerlendirmesi başlıyor...")
        
        metrics = {}
        
        # Perplexity
        print("📊 Perplexity hesaplanıyor...")
        metrics['perplexity'] = self.calculate_perplexity(dataloader)
        
        # Accuracy
        print("📊 Accuracy hesaplanıyor...")
        metrics['accuracy'] = self.calculate_accuracy(dataloader)
        
        # BLEU (eğer test prompt'ları verildiyse)
        if test_prompts:
            print("📊 BLEU skoru hesaplanıyor...")
            # Örnek cevaplar üret
            generated_responses = self.generate_sample_responses(test_prompts)
            
            # Referans cevaplar (gerçek hayatta veri setinden gelir)
            # Şimdilik basit bir örnek
            reference_responses = [f"Bu bir test cevabıdır: {prompt}" for prompt in test_prompts]
            
            bleu_scores = self.calculate_bleu(generated_responses, reference_responses)
            metrics.update(bleu_scores)
        
        print("✅ Model değerlendirmesi tamamlandı!")
        return metrics
    
    def save_evaluation_results(self, results: Dict[str, float], filepath: str):
        """
        Değerlendirme sonuçlarını kaydet
        
        Args:
            results: Değerlendirme sonuçları
            filepath: Kayıt dosya yolu
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Değerlendirme sonuçları kaydedildi: {filepath}")


def calculate_perplexity(model, dataloader) -> float:
    """Standalone perplexity hesaplama fonksiyonu"""
    evaluator = ModelEvaluator(model, None, device=next(model.parameters()).device)
    return evaluator.calculate_perplexity(dataloader)


def calculate_bleu(model, dataset, tokenizer) -> Dict[str, float]:
    """Standalone BLEU hesaplama fonksiyonu"""
    evaluator = ModelEvaluator(model, tokenizer, device=next(model.parameters()).device)
    
    # Basit test prompt'ları
    test_prompts = [
        "Merhaba, nasılsın?",
        "Hava nasıl?",
        "Python nedir?",
        "Makine öğrenmesi hakkında bilgi ver"
    ]
    
    generated_responses = evaluator.generate_sample_responses(test_prompts)
    reference_responses = [f"Test cevabı: {prompt}" for prompt in test_prompts]
    
    return evaluator.calculate_bleu(generated_responses, reference_responses)


if __name__ == "__main__":
    # Test
    print("🧪 Evaluation modülü test ediliyor...")
    print("✅ Modül başarıyla yüklendi!")

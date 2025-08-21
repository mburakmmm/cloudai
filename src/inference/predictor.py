"""
Text Generation Predictor
Eğitilmiş model ile metin üretimi yapan sınıf
"""

import os
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Union
import numpy as np


class Predictor:
    """Eğitilmiş model ile text generation yapan sınıf"""
    
    def __init__(self, model_path: str, tokenizer_path: str, config: Optional[Dict] = None):
        """
        Predictor'ı başlat
        
        Args:
            model_path: Eğitilmiş model dosya yolu
            tokenizer_path: Eğitilmiş tokenizer dosya yolu
            config: Konfigürasyon parametreleri
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config or {}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Predictor cihazı: {self.device}")
        
        # Model ve tokenizer'ı yükle
        self.model = None
        self.tokenizer = None
        
        self._load_model_and_tokenizer()
        
        # Generation parametreleri
        self.default_params = {
            'max_length': 100,
            'temperature': 0.3,  # 0.7 → 0.3 (daha deterministik)
            'top_k': 20,  # 50 → 20 (daha odaklı)
            'top_p': 0.8,  # 0.9 → 0.8 (daha kontrollü)
            'do_sample': True,
            'num_beams': 3,  # 1 → 3 (beam search)
            'repetition_penalty': 1.2,  # 1.1 → 1.2 (tekrarı azalt)
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3
        }
        
        print(f"✅ Predictor hazır! Model: {os.path.basename(model_path)}")
    
    def _load_model_and_tokenizer(self):
        """Model ve tokenizer'ı yükle"""
        try:
            # Tokenizer yükle
            from src.model.tokenizer import CustomTokenizer
            self.tokenizer = CustomTokenizer()
            self.tokenizer.load(self.tokenizer_path)
            print(f"📂 Tokenizer yüklendi: {self.tokenizer_path}")
            
            # Model yükle
            from src.model.transformer_model import GenerativeTransformer
            model_config = self.config.get('model', {})
            
            # Model'i oluştur
            self.model = GenerativeTransformer(
                vocab_size=self.tokenizer.get_vocab_size(),
                d_model=model_config.get('d_model', 512),
                nhead=model_config.get('nhead', 8),
                num_decoder_layers=model_config.get('num_decoder_layers', 6),
                dim_feedforward=model_config.get('dim_feedforward', 2048),
                dropout=0.0  # Inference'da dropout kapalı
            )
            
            # Eğitilmiş ağırlıkları yükle
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Model'i eval moduna al ve device'a taşı
            self.model.eval()
            self.model.to(self.device)
            
            print(f"🧠 Model yüklendi: {self.model_path}")
            print(f"📊 Model parametreleri: {self.model.count_parameters():,}")
            
        except Exception as e:
            print(f"❌ Model/Tokenizer yükleme hatası: {e}")
            raise
    
    def generate_response(self, prompt_text: str, max_length: Optional[int] = None, 
                         strategy: str = 'greedy', conversation_history: Optional[List[Dict]] = None, **kwargs) -> str:
        """
        Prompt'a cevap üret
        
        Args:
            prompt_text: Girdi metni
            max_length: Maksimum üretim uzunluğu
            strategy: Üretim stratejisi ('greedy', 'beam', 'top-k', 'nucleus')
            conversation_history: Konuşma geçmişi [{'role': 'user', 'content': '...'}, {'role': 'bot', 'content': '...'}]
            **kwargs: Diğer generation parametreleri
            
        Returns:
            str: Üretilen cevap
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model veya tokenizer yüklenmemiş!")
        
        # Konuşma geçmişini prompt'a ekle
        enriched_prompt = self._build_contextual_prompt(prompt_text, conversation_history)
        
        # Stratejiye göre parametreleri ayarla
        generation_params = self._get_generation_params(strategy, max_length, **kwargs)
        
        try:
            # Zenginleştirilmiş prompt'u tokenize et
            encoding = self.tokenizer.encode(enriched_prompt, max_length=512)
            input_ids = encoding['input_ids'].to(self.device)
            
            # Generation
            with torch.no_grad():
                # Transformer model'in generate metodu için parametreleri uyarla
                generate_kwargs = {
                    'max_length': generation_params.get('max_length', 50),
                    'temperature': generation_params.get('temperature', 0.7),
                    'top_k': generation_params.get('top_k', 50),
                    'top_p': generation_params.get('top_p', 0.9),
                    'num_beams': generation_params.get('num_beams', 1),
                    'repetition_penalty': generation_params.get('repetition_penalty', 1.1)
                }
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    **generate_kwargs
                )
            
            # Sadece yeni üretilen kısmı al
            new_tokens = generated_ids[0, input_ids.size(0):]
            
            # Metne çevir
            response = self.tokenizer.decode(new_tokens.tolist())
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Generation hatası: {e}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"
    
    def _get_generation_params(self, strategy: str, max_length: Optional[int], **kwargs) -> Dict:
        """
        Üretim stratejisine göre parametreleri ayarla
        
        Args:
            strategy: Üretim stratejisi
            max_length: Maksimum uzunluk
            **kwargs: Ek parametreler
            
        Returns:
            Dict: Generation parametreleri
        """
        # Base parametreler
        base_params = {
            'max_length': max_length or self.default_params.get('max_length', 50),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'do_sample': False
        }
        
        if strategy == 'greedy':
            # Basit greedy search
            return {**base_params, **self.default_params}
        
        elif strategy == 'beam':
            # Beam search
            return {
                **base_params,
                'do_sample': False,
                'num_beams': kwargs.get('num_beams', 5),
                'early_stopping': kwargs.get('early_stopping', True),
                'length_penalty': kwargs.get('length_penalty', 1.0),
                'no_repeat_ngram_size': kwargs.get('no_repeat_ngram_size', 3)
            }
        
        elif strategy == 'top-k':
            # Top-K sampling
            return {
                **base_params,
                'do_sample': True,
                'top_k': kwargs.get('top_k', 50),
                'temperature': kwargs.get('temperature', 0.7),
                'repetition_penalty': kwargs.get('repetition_penalty', 1.1)
            }
        
        elif strategy == 'nucleus':
            # Nucleus (Top-P) sampling
            return {
                **base_params,
                'do_sample': True,
                'top_p': kwargs.get('top_p', 0.9),
                'temperature': kwargs.get('temperature', 0.8),
                'repetition_penalty': kwargs.get('repetition_penalty', 1.1)
            }
        
        else:
            print(f"⚠️ Bilinmeyen strateji: {strategy}. Greedy kullanılıyor.")
            return {**base_params, **self.default_params}
    
    def _build_contextual_prompt(self, prompt_text: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Konuşma geçmişi ile prompt'u zenginleştir
        
        Args:
            prompt_text: Mevcut prompt
            conversation_history: Konuşma geçmişi
            
        Returns:
            str: Zenginleştirilmiş prompt
        """
        if not conversation_history:
            return prompt_text
        
        # Konuşma geçmişini birleştir
        context_parts = []
        for turn in conversation_history[-5:]:  # Son 5 tur
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                context_parts.append(f"Kullanıcı: {content}")
            elif role == 'bot':
                context_parts.append(f"Bot: {content}")
        
        # Mevcut prompt'u ekle
        context_parts.append(f"Kullanıcı: {prompt_text}")
        
        # Tümünü birleştir
        enriched_prompt = "\n".join(context_parts)
        
        return enriched_prompt
    
    def generate_multiple_responses(self, prompt_text: str, num_responses: int = 3, 
                                  strategy: str = 'nucleus', **kwargs) -> List[str]:
        """
        Aynı prompt için birden fazla cevap üret
        
        Args:
            prompt_text: Giriş metni
            num_responses: Üretilecek cevap sayısı
            strategy: Üretim stratejisi
            **kwargs: Ek parametreler
            
        Returns:
            List[str]: Üretilen cevaplar
        """
        responses = []
        
        for i in range(num_responses):
            # Her seferinde farklı seed kullan
            if 'temperature' in kwargs:
                kwargs['temperature'] = kwargs['temperature'] + (i * 0.1)
            
            response = self.generate_response(prompt_text, strategy=strategy, **kwargs)
            responses.append(response)
        
        return responses
    
    def generate_conversation(self, prompt: str, max_turns: int = 3, **kwargs) -> List[Dict]:
        """
        Çok tur konuşma üret
        
        Args:
            prompt: Başlangıç prompt'u
            max_turns: Maksimum konuşma turu
            **kwargs: Generation parametreleri
            
        Returns:
            List[Dict]: Konuşma geçmişi
        """
        conversation = []
        current_prompt = prompt
        
        for turn in range(max_turns):
            # Cevap üret
            response = self.generate_response(current_prompt, **kwargs)
            
            # Konuşmaya ekle
            conversation.append({
                'turn': turn + 1,
                'prompt': current_prompt,
                'response': response,
                'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
            })
            
            # Sonraki prompt için response'u ekle
            current_prompt = f"{current_prompt} {response}"
            
            # Çok uzun olmasın
            if len(current_prompt.split()) > 200:
                break
        
        return conversation
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Birden fazla prompt için batch generation
        
        Args:
            prompts: Prompt listesi
            **kwargs: Generation parametreleri
            
        Returns:
            List[str]: Cevap listesi
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            print(f"🔄 Generating {i+1}/{len(prompts)}...")
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def get_generation_stats(self) -> Dict:
        """Generation istatistiklerini döndür"""
        if not self.model:
            return {}
        
        return {
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'vocab_size': self.tokenizer.get_vocab_size() if self.tokenizer else 0,
            'model_parameters': self.model.count_parameters(),
            'device': str(self.device),
            'default_params': self.default_params
        }
    
    def update_generation_params(self, **kwargs):
        """Generation parametrelerini güncelle"""
        self.default_params.update(kwargs)
        print(f"🔧 Generation parametreleri güncellendi: {kwargs}")
    
    def test_generation(self, test_prompts: Optional[List[str]] = None) -> Dict:
        """Generation'ı test et"""
        if test_prompts is None:
            test_prompts = [
                "Merhaba, nasılsın?",
                "Python programlama hakkında bilgi verir misin?",
                "Yapay zeka nedir?",
                "Günün nasıl geçiyor?"
            ]
        
        results = {}
        
        for prompt in test_prompts:
            print(f"\n🧪 Test: {prompt}")
            try:
                response = self.generate_response(prompt, max_length=50)
                results[prompt] = {
                    'success': True,
                    'response': response,
                    'length': len(response.split())
                }
                print(f"✅ Cevap: {response}")
            except Exception as e:
                results[prompt] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ Hata: {e}")
        
        return results


def create_predictor_from_checkpoint(checkpoint_path: str, tokenizer_path: str, 
                                   config: Optional[Dict] = None) -> Predictor:
    """
    Checkpoint'ten predictor oluştur
    
    Args:
        checkpoint_path: Checkpoint dosya yolu
        tokenizer_path: Tokenizer dosya yolu
        config: Konfigürasyon
        
    Returns:
        Predictor: Oluşturulan predictor
    """
    # Checkpoint'ten model path'i al
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            if config:
                config.update(checkpoint_config)
            else:
                config = checkpoint_config
    
    return Predictor(checkpoint_path, tokenizer_path, config)


if __name__ == "__main__":
    # Test
    try:
        predictor = Predictor(
            model_path="models/my_chatbot.pth",
            tokenizer_path="models/tokenizer"
        )
        
        # Test generation
        test_results = predictor.test_generation()
        print(f"\n📊 Test Sonuçları: {test_results}")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")

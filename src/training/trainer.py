"""
Model Training Implementation
PyTorch ile model eğitimi için Trainer sınıfı
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    """Model eğitimi için Trainer sınıfı"""
    
    def __init__(self, model: nn.Module, dataset, config: Dict):
        """
        Trainer'ı başlat
        
        Args:
            model: Eğitilecek model
            dataset: Eğitim veri seti
            config: Eğitim konfigürasyonu
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Eğitim cihazı: {self.device}")
        
        # Model'i device'a taşı
        self.model.to(self.device)
        
        # Mixed Precision (FP16) - 2x hız
        self.use_amp = self.config.get('mixed_precision', False) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🚀 Mixed Precision (FP16) etkinleştirildi!")
        
        # Model Compile (PyTorch 2.0+) - 1.2-1.5x hız
        if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("🚀 Model compile edildi (PyTorch 2.0+)!")
            except Exception as e:
                print(f"⚠️ Model compile hatası: {e}")
        
        # Eğitim durumu
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        
        # Optimizer ve loss
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Setup
        self._setup_optimizer_and_loss()
        self._setup_scheduler()
        
        # Checkpoint dizini
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"✅ Trainer hazır! Model parametreleri: {self.model.count_parameters():,}")
        if self.use_amp:
            print(f"🚀 Mixed Precision: Etkin (FP16)")
        print(f"🚀 Model Compile: {'Etkin' if hasattr(self.model, '_orig_mod') else 'Devre dışı'}")
    
    def _setup_optimizer_and_loss(self):
        """Optimizer ve loss fonksiyonunu ayarla"""
        # Optimizer
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"🔧 Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"🔧 Loss: CrossEntropyLoss (ignore_index=-100)")
    
    def _setup_scheduler(self):
        """Learning rate scheduler'ı ayarla"""
        # Cosine annealing scheduler
        num_epochs = self.config.get('num_epochs', 10)
        warmup_steps = self.config.get('warmup_steps', 100)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=num_epochs,
            T_mult=2,
            eta_min=1e-6
        )
        
        print(f"🔧 Scheduler: CosineAnnealingWarmRestarts (T_0={num_epochs})")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Tek epoch eğitimi"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        # Gradient accumulation
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Batch'i device'a taşı
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed Precision Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1)
                    )
                    loss = loss / accumulation_steps  # Normalize loss
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                loss = loss / accumulation_steps  # Normalize loss
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Loss'u topla
            total_loss += loss.item() * accumulation_steps
            
            # Progress (daha az sıklıkta)
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item() * accumulation_steps:.4f}")
        
        # Scheduler step
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Model değerlendirmesi"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # Batch'i device'a taşı
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Loss hesapla
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Checkpoint kaydet"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Normal checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # En iyi checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 En iyi model kaydedildi: {best_path}")
        
        print(f"💾 Checkpoint kaydedildi: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Checkpoint yükle"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        
        print(f"📂 Checkpoint yüklendi: {checkpoint_path}")
        print(f"📊 Epoch: {self.current_epoch}, Loss: {self.best_loss:.4f}")
        
        return True
    
    def train(self, num_epochs: Optional[int] = None, 
              batch_size: int = 16, 
              validation_split: float = 0.1,
              early_stopping_patience: int = 5,
              save_every: int = 1) -> Dict:
        """
        Model eğitimi
        
        Args:
            num_epochs: Eğitim epoch sayısı
            batch_size: Batch boyutu
            validation_split: Validation için ayrılacak oran
            early_stopping_patience: Early stopping sabır değeri
            save_every: Kaç epoch'ta bir kaydet
            
        Returns:
            Dict: Eğitim sonuçları
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 10)
        
        print(f"🚀 Eğitim başlıyor... {num_epochs} epoch")
        print(f"📊 Batch size: {batch_size}, Validation split: {validation_split}")
        print(f"🚀 Mixed Precision: {'Etkin' if self.use_amp else 'Devre dışı'}")
        print(f"🚀 Model Compile: {'Etkin' if hasattr(self.model, '_orig_mod') else 'Devre dışı'}")
        print(f"📈 Gradient Accumulation: {self.config.get('gradient_accumulation_steps', 1)}")
        print(f"🔧 Effective Batch Size: {batch_size * self.config.get('gradient_accumulation_steps', 1)}")
        
        # DataLoader oluştur
        train_size = int((1 - validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # DataLoader optimizasyonları
        dataloader_config = self.config.get('dataloader', {})
        hardware_config = self.config.get('hardware', {})
        
        # CUDA varlığına göre pin_memory ayarla
        pin_memory = torch.cuda.is_available() and hardware_config.get('pin_memory', True)
        num_workers = hardware_config.get('num_workers', 4) if torch.cuda.is_available() else 0
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=dataloader_config.get('persistent_workers', True) if num_workers > 0 else False,
            prefetch_factor=dataloader_config.get('prefetch_factor', 2) if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=dataloader_config.get('persistent_workers', True) if num_workers > 0 else False,
            prefetch_factor=dataloader_config.get('prefetch_factor', 2) if num_workers > 0 else None
        )
        
        print(f"📚 Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        # Eğitim döngüsü
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch + 1
            
            print(f"\n🔄 Epoch {self.current_epoch}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.training_losses.append(train_loss)
            
            # Validation
            val_loss = self.evaluate(val_loader)
            self.validation_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"✅ Epoch {self.current_epoch} tamamlandı")
            print(f"📊 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"⏱️  Süre: {epoch_time:.2f}s")
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"📈 Learning Rate: {current_lr:.2e}")
            
            # Checkpoint kaydet
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping! {early_stopping_patience} epoch boyunca iyileşme yok")
                break
        
        # Eğitim tamamlandı
        total_time = time.time() - start_time
        print(f"\n🎉 Eğitim tamamlandı!")
        print(f"⏱️  Toplam süre: {total_time:.2f}s")
        print(f"🏆 En iyi validation loss: {self.best_loss:.4f}")
        
        # Sonuçları döndür
        results = {
            'final_train_loss': self.training_losses[-1] if self.training_losses else None,
            'final_val_loss': self.validation_losses[-1] if self.validation_losses else None,
            'best_val_loss': self.best_loss,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'total_time': total_time,
            'epochs_trained': len(self.training_losses)
        }
        
        return results
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Eğitim eğrilerini çiz"""
        if not self.training_losses:
            print("❌ Henüz eğitim yapılmamış!")
            return
        
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.training_losses) + 1)
        
        plt.plot(epochs, self.training_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.validation_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict:
        """Eğitim özeti"""
        return {
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'total_parameters': self.model.count_parameters(),
            'device': str(self.device),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }

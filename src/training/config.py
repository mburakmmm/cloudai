"""
Training Configuration
Eğitim için gerekli tüm konfigürasyon parametreleri
"""

import os
from typing import Dict, Any


def get_model_config() -> Dict[str, Any]:
    """Model konfigürasyonu"""
    return {
        "vocab_size": 8000,  # 30000 → 8000 (daha iyi öğrenme)
        "d_model": 512,
        "nhead": 8,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "max_seq_length": 512,
        "dropout": 0.1
    }


def get_training_config() -> Dict[str, Any]:
    """Eğitim konfigürasyonu"""
    return {
        "learning_rate": 1e-4,  # 2e-4 → 1e-4 (daha stabil)
        "weight_decay": 0.01,
        "batch_size": 16,  # 32 → 16 (daha iyi öğrenme)
        "num_epochs": 50,  # 10 → 50 (5x daha fazla eğitim)
        "max_seq_length": 256,
        "warmup_steps": 500,  # 100 → 500 (daha uzun warmup)
        "gradient_clip_val": 1.0,
        "early_stopping_patience": 10,  # 5 → 10 (daha sabırlı)
        "save_every": 5,  # 1 → 5 (daha az checkpoint)
        "gradient_accumulation_steps": 4,  # 2 → 4 (daha büyük effective batch)
        "eval_every": 5  # 2 → 5 (daha az validation)
    }


def get_paths_config() -> Dict[str, Any]:
    """Dosya yolları konfigürasyonu"""
    return {
        "data": None,  # Veritabanından çekilecek
        "model_save_path": "models/my_chatbot.pth",
        "tokenizer_path": "models/tokenizer",
        "checkpoint_dir": "checkpoints",
        "logs_dir": "logs",
        "output_dir": "outputs"
    }


def get_hardware_config() -> Dict[str, Any]:
    """Donanım konfigürasyonu"""
    return {
        "device": "auto",  # auto, cuda, cpu, mps
        "num_workers": 8,  # 4 → 8 (2x paralel işlem)
        "pin_memory": True,
        "mixed_precision": True,  # False → True (FP16, 2x hız)
        "compile_model": True  # Yeni: PyTorch 2.0+ compile
    }


def get_optimizer_config() -> Dict[str, Any]:
    """Optimizer konfigürasyonu"""
    return {
        "type": "adamw",  # adamw, adam, sgd
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False
    }


def get_scheduler_config() -> Dict[str, Any]:
    """Scheduler konfigürasyonu"""
    return {
        "type": "cosine_with_warmup",  # cosine_with_warmup, onecycle, step
        "warmup_steps": 100,
        "min_lr": 1e-6
    }


def get_dataloader_config() -> Dict[str, Any]:
    """DataLoader konfigürasyonu"""
    return {
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,  # Yeni: Worker'ları yeniden başlatma
        "prefetch_factor": 2  # Yeni: Veri önceden yükleme
    }


def get_validation_config() -> Dict[str, Any]:
    """Validation konfigürasyonu"""
    return {
        "val_split": 0.1,
        "eval_batch_size": 16,
        "eval_every": 1
    }


def get_logging_config() -> Dict[str, Any]:
    """Logging konfigürasyonu"""
    return {
        "log_every_n_steps": 10,
        "save_every_n_epochs": 1,
        "tensorboard": False,
        "wandb": False
    }


def get_config() -> Dict[str, Any]:
    """Ana konfigürasyon"""
    return {
        "model": get_model_config(),
        "training": get_training_config(),
        "paths": get_paths_config(),
        "hardware": get_hardware_config(),
        "optimizer": get_optimizer_config(),
        "scheduler": get_scheduler_config(),
        "dataloader": get_dataloader_config(),
        "validation": get_validation_config(),
        "logging": get_logging_config()
    }


def create_directories():
    """Gerekli dizinleri oluştur"""
    config = get_config()
    
    dirs_to_create = [
        config["paths"]["checkpoint_dir"],
        config["paths"]["logs_dir"],
        config["paths"]["output_dir"],
        os.path.dirname(config["paths"]["model_save_path"]),
        config["paths"]["tokenizer_path"]
    ]
    
    for dir_path in dirs_to_create:
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Dizin oluşturuldu: {dir_path}")


def print_config_summary():
    """Konfigürasyon özetini yazdır"""
    config = get_config()
    
    print("🔧 Konfigürasyon Özeti:")
    print("=" * 50)
    
    print(f"📊 Model: {config['model']['d_model']}D, {config['model']['num_decoder_layers']} layers")
    print(f"🎯 Eğitim: {config['training']['num_epochs']} epoch, lr={config['training']['learning_rate']}")
    print(f"📦 Batch: {config['training']['batch_size']}, Seq Len: {config['training']['max_seq_length']}")
    print(f"💾 Kayıt: {config['paths']['model_save_path']}")
    print(f"🔧 Optimizer: {config['optimizer']['type'].upper()}")
    print(f"📈 Scheduler: {config['scheduler']['type']}")
    print(f"💻 Cihaz: {config['hardware']['device']}")


if __name__ == "__main__":
    # Test
    create_directories()
    print_config_summary()

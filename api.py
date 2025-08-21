"""
Cloud AI REST API
FastAPI ile model servis API'si
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import asyncio
import threading
import time
import json
import os
import sys

# Proje root'unu path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.inference.predictor import Predictor
    from src.training.config import get_model_config
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Predictor import hatası: {e}")
    PREDICTOR_AVAILABLE = False

# FastAPI app
app = FastAPI(
    title="Cloud AI API",
    description="PyTorch Transformer tabanlı AI Chatbot API'si",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic modelleri
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Giriş metni", min_length=1, max_length=1000)
    max_length: Optional[int] = Field(50, description="Maksimum üretim uzunluğu", ge=10, le=200)
    strategy: str = Field("greedy", description="Üretim stratejisi", regex="^(greedy|beam|top-k|nucleus)$")
    temperature: Optional[float] = Field(0.7, description="Sampling sıcaklığı", ge=0.1, le=2.0)
    num_beams: Optional[int] = Field(5, description="Beam search için beam sayısı", ge=2, le=10)
    top_k: Optional[int] = Field(50, description="Top-K sampling için K değeri", ge=10, le=100)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling için P değeri", ge=0.1, le=1.0)

class GenerateResponse(BaseModel):
    response: str = Field(..., description="Üretilen cevap")
    strategy: str = Field(..., description="Kullanılan strateji")
    generation_time: float = Field(..., description="Üretim süresi (saniye)")
    tokens_generated: int = Field(..., description="Üretilen token sayısı")
    model_info: Dict = Field(..., description="Model bilgileri")

class MultipleGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Giriş metni", min_length=1, max_length=1000)
    num_responses: int = Field(3, description="Üretilecek cevap sayısı", ge=1, le=10)
    strategy: str = Field("nucleus", description="Üretim stratejisi")
    **kwargs: Optional[Dict] = Field({}, description="Ek parametreler")

class MultipleGenerateResponse(BaseModel):
    responses: List[str] = Field(..., description="Üretilen cevaplar")
    strategy: str = Field(..., description="Kullanılan strateji")
    total_time: float = Field(..., description="Toplam üretim süresi")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API durumu")
    model_loaded: bool = Field(..., description="Model yüklenmiş mi")
    model_info: Optional[Dict] = Field(None, description="Model bilgileri")
    timestamp: str = Field(..., description="Kontrol zamanı")

# Global predictor instance
predictor = None
model_loading = False

def load_predictor():
    """Predictor'ı yükle (background thread'de)"""
    global predictor, model_loading
    
    try:
        print("🚀 Model yükleniyor...")
        model_loading = True
        
        # Model konfigürasyonu
        model_config = get_model_config()
        
        # Predictor oluştur
        predictor = Predictor(
            model_path="models/my_chatbot.pth",
            tokenizer_path="models/tokenizer",
            config={'model': model_config}
        )
        
        print("✅ Model başarıyla yüklendi!")
        model_loading = False
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        model_loading = False
        predictor = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında çalışır"""
    print("🚀 Cloud AI API başlatılıyor...")
    
    # Model yükleme thread'ini başlat
    if PREDICTOR_AVAILABLE:
        threading.Thread(target=load_predictor, daemon=True).start()
    else:
        print("⚠️ Predictor kullanılamıyor!")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API sağlık kontrolü"""
    global predictor, model_loading
    
    status = "healthy"
    if not PREDICTOR_AVAILABLE:
        status = "unavailable"
    elif model_loading:
        status = "loading"
    elif predictor is None:
        status = "no_model"
    
    model_info = None
    if predictor and predictor.model:
        model_info = {
            "parameters": predictor.model.count_parameters(),
            "device": str(predictor.device),
            "vocab_size": predictor.tokenizer.get_vocab_size() if predictor.tokenizer else None
        }
    
    return HealthResponse(
        status=status,
        model_loaded=predictor is not None,
        model_info=model_info,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# Ana generation endpoint
@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Tek cevap üret"""
    global predictor
    
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Predictor kullanılamıyor")
    
    if predictor is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model yükleniyor, lütfen bekleyin")
        else:
            raise HTTPException(status_code=503, detail="Model yüklenmemiş")
    
    try:
        start_time = time.time()
        
        # Generation parametreleri
        kwargs = {}
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        if request.strategy == 'beam' and request.num_beams:
            kwargs['num_beams'] = request.num_beams
        if request.strategy == 'top-k' and request.top_k:
            kwargs['top_k'] = request.top_k
        if request.strategy == 'nucleus' and request.top_p:
            kwargs['top_p'] = request.top_p
        
        # Cevap üret
        response = predictor.generate_response(
            prompt_text=request.prompt,
            max_length=request.max_length,
            strategy=request.strategy,
            **kwargs
        )
        
        generation_time = time.time() - start_time
        
        # Token sayısını hesapla (basit yaklaşım)
        tokens_generated = len(response.split())
        
        # Model bilgileri
        model_info = {
            "parameters": predictor.model.count_parameters(),
            "device": str(predictor.device),
            "vocab_size": predictor.tokenizer.get_vocab_size() if predictor.tokenizer else None
        }
        
        return GenerateResponse(
            response=response,
            strategy=request.strategy,
            generation_time=generation_time,
            tokens_generated=tokens_generated,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation hatası: {str(e)}")

# Çoklu cevap üretimi
@app.post("/generate-multiple", response_model=MultipleGenerateResponse)
async def generate_multiple_responses(request: MultipleGenerateRequest):
    """Birden fazla cevap üret"""
    global predictor
    
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Predictor kullanılamıyor")
    
    if predictor is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model yükleniyor, lütfen bekleyin")
        else:
            raise HTTPException(status_code=503, detail="Model yüklenmemiş")
    
    try:
        start_time = time.time()
        
        # Çoklu cevap üret
        responses = predictor.generate_multiple_responses(
            prompt_text=request.prompt,
            num_responses=request.num_responses,
            strategy=request.strategy,
            **request.kwargs
        )
        
        total_time = time.time() - start_time
        
        return MultipleGenerateResponse(
            responses=responses,
            strategy=request.strategy,
            total_time=total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation hatası: {str(e)}")

# Model yeniden yükleme
@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Modeli yeniden yükle"""
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Predictor kullanılamıyor")
    
    background_tasks.add_task(load_predictor)
    
    return {"message": "Model yeniden yükleniyor", "status": "reloading"}

# Model bilgileri
@app.get("/model-info")
async def get_model_info():
    """Model bilgilerini getir"""
    global predictor
    
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Predictor kullanılamıyor")
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model yüklenmemiş")
    
    try:
        model_info = {
            "parameters": predictor.model.count_parameters(),
            "device": str(predictor.device),
            "vocab_size": predictor.tokenizer.get_vocab_size() if predictor.tokenizer else None,
            "model_path": predictor.model_path,
            "tokenizer_path": predictor.tokenizer_path
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model bilgisi alınamadı: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Ana sayfa"""
    return {
        "message": "🚀 Cloud AI API",
        "description": "PyTorch Transformer tabanlı AI Chatbot API'si",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate": "/generate",
            "generate_multiple": "/generate-multiple",
            "model_info": "/model-info",
            "reload_model": "/reload-model"
        }
    }

if __name__ == "__main__":
    print("🚀 Cloud AI API başlatılıyor...")
    print("📖 API dokümantasyonu: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

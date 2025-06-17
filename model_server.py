from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json

app = FastAPI(
    title="VQA RAG Model Server",
    description="GPU 전용 모델 추론 서버",
    version="1.0.0"
)

# 전역 모델 변수
embedder = None
text_model = None
text_tokenizer = None
vqa_model = None
vqa_tokenizer = None
vqa_preprocessor = None

# 요청/응답 모델들
class EmbedRequest(BaseModel):
    texts: List[str]
    convert_to_numpy: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    message: str = "success"

class TextGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    system: Optional[str] = None
    kv_cache: Optional[Dict] = None

class TextGenerateResponse(BaseModel):
    response: str
    message: str = "success"

class VQAGenerateRequest(BaseModel):
    prompt: str
    image_base64: str  # base64 인코딩된 이미지
    max_new_tokens: int = 256

class VQAGenerateResponse(BaseModel):
    response: str
    message: str = "success"

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    gpu_info: Dict[str, Any]

def load_models():
    """모델들을 로드하는 함수"""
    global embedder, text_model, text_tokenizer, vqa_model, vqa_tokenizer, vqa_preprocessor
    
    try:
        # 임베더 로드
        embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 텍스트 모델 로드
        text_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        text_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name, 
            device_map={"": 1}, 
            torch_dtype=torch.float16, 
            quantization_config=text_bnb_config
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name, 
            padding_side='left', 
            use_fast=True
        )
        
        # VQA 모델 로드 - GPU 3 사용
        vqa_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        vqa_model = AutoModelForCausalLM.from_pretrained(
            vqa_model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16, 
            device_map={"": 3}
        )
        vqa_preprocessor = AutoProcessor.from_pretrained(vqa_model_name, trust_remote_code=True)
        vqa_tokenizer = AutoTokenizer.from_pretrained(vqa_model_name, padding_side='left', use_fast=True)
        
    except Exception as e:
        raise e

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_models()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    models_loaded = {
        "embedder": embedder is not None,
        "text_model": text_model is not None,
        "text_tokenizer": text_tokenizer is not None,
        "vqa_model": vqa_model is not None,
        "vqa_tokenizer": vqa_tokenizer is not None,
        "vqa_preprocessor": vqa_preprocessor is not None
    }
    
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i)
            }
    
    return HealthResponse(
        status="healthy" if all(models_loaded.values()) else "unhealthy",
        models_loaded=models_loaded,
        gpu_info=gpu_info
    )

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """텍스트 임베딩 생성"""
    if embedder is None:
        raise HTTPException(status_code=500, detail="임베더가 로드되지 않았습니다.")
    
    try:
        embeddings = embedder.encode(request.texts, convert_to_numpy=request.convert_to_numpy)
        
        # numpy array를 list로 변환
        if request.convert_to_numpy:
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [emb.tolist() for emb in embeddings]
        
        return EmbedResponse(embeddings=embeddings_list)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 생성 오류: {str(e)}")

@app.post("/text/generate", response_model=TextGenerateResponse)
async def generate_text(request: TextGenerateRequest):
    """텍스트 생성"""
    if text_model is None or text_tokenizer is None:
        raise HTTPException(status_code=500, detail="텍스트 모델이 로드되지 않았습니다.")
    
    try:
        from utils import hf_generate_response
        
        response = hf_generate_response(
            text_model, 
            text_tokenizer, 
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            system=request.system,
            kv_cache=request.kv_cache
        )
        
        return TextGenerateResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 오류: {str(e)}")

@app.post("/vqa/generate", response_model=VQAGenerateResponse)
async def generate_vqa(request: VQAGenerateRequest):
    """VQA 생성"""
    if vqa_model is None or vqa_tokenizer is None or vqa_preprocessor is None:
        raise HTTPException(status_code=500, detail="VQA 모델이 로드되지 않았습니다.")
    
    try:
        # base64 이미지 디코딩
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))
        
        from utils import vqa_generate_response
        
        response = vqa_generate_response(
            vqa_model,
            vqa_tokenizer,
            vqa_preprocessor,
            request.prompt,
            image,
            max_new_tokens=request.max_new_tokens
        )
        
        return VQAGenerateResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VQA 생성 오류: {str(e)}")

@app.get("/models")
async def list_models():
    """로드된 모델 목록 조회"""
    return {
        "models": {
            "embedder": "jhgan/ko-sroberta-multitask",
            "text_model": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
            "vqa_model": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        },
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
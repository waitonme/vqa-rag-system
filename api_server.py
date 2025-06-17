from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import fitz
from datetime import datetime
import json
import uuid
import time
import asyncio
import numpy as np
from utils import simple_parse
from models_client import ModelServerClient, VQARAGModelWrapper
from debug_chunk import print_chunk_debug, detect_debug_command
from io import BytesIO
from PIL import Image
import base64
import requests as req

app = FastAPI(
    title="VQA RAG API Server",
    description="OpenWebUI 호환 PDF 처리 및 RAG API 서버 (모델 서버 분리)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 인증 스키마
security = HTTPBearer()

# 간단한 API 키 검증
VALID_API_KEYS = {"test-api-key", "your-api-key-here"}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """API 키 검증"""
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# 모델 서버 클라이언트
model_client = None

# 현재 활성화된 모델과 파일 정보
uploaded_files = {}
file_models = {}

# Validation 오류 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": "Validation error occurred"}
    )

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"⏱️ {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    return response

class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    images: Optional[Any] = None
    image_url: Optional[Dict[str, Any]] = None

class FileReference(BaseModel):
    type: str
    id: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    files: Optional[List[FileReference]] = []
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    features: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    stream_options: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    images: Optional[Any] = None
    image_urls: Optional[Any] = None
    
    class Config:
        extra = "allow"

class ChatResponse(BaseModel):
    id: Optional[str] = None
    object: str = "chat.completion"
    created: Optional[int] = None
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vqa-rag"

class FileUploadResponse(BaseModel):
    id: str
    user_id: str = "system"
    hash: Optional[str] = None
    filename: str
    data: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any]
    created_at: int
    updated_at: int

def connect_to_model_server():
    """모델 서버 연결 초기화"""
    global model_client
    
    model_client = ModelServerClient("http://localhost:8001")
    
    if not model_client.is_healthy():
        print("모델 서버 연결 실패! 모델 서버를 먼저 시작해주세요.")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 서버 연결"""
    connect_to_model_server()

@app.get("/v1/models")
async def list_models():
    """사용 가능한 모델 목록 반환"""
    base_models = [
        "text_only_rag", "vqa_rag", "vqa_rag_cag", "async_vqa_rag_cag"
    ]
    
    model_list = []
    for model_id in base_models:
        model_list.append(ModelInfo(
            id=model_id,
            created=int(datetime.now().timestamp())
        ))
    
    return {"data": model_list}

@app.get("/api/models")
async def list_models_api(api_key: str = Depends(verify_api_key)):
    """OpenWebUI 표준 모델 목록 엔드포인트"""
    return await list_models()

@app.get("/api/v1/models")
async def list_models_v1(api_key: str = Depends(verify_api_key)):
    """OpenWebUI v1 모델 목록 엔드포인트"""
    return await list_models_api(api_key)

async def extract_pdf_content(temp_file_path: str):
    """PDF 내용 추출 (텍스트와 이미지)"""
    try:
        pdf = fitz.open(temp_file_path)
        simple_texts, simple_images = simple_parse(pdf)
        return pdf, simple_texts, simple_images
        
    except Exception as e:
        raise e

async def create_file_model(file_id: str, pdf, simple_texts: List[str], simple_images: List, clean_filename: str):
    """파일별 모델 생성 및 텍스트 청킹"""
    global model_client, file_models
    
    if not model_client or not model_client.is_healthy():
        raise HTTPException(status_code=500, detail="모델 서버에 연결할 수 없습니다.")
    
    # 파일별 모델 래퍼 생성
    file_model = VQARAGModelWrapper(model_client)
    
    # 모델 데이터 설정
    file_model.pdf = pdf
    file_model.texts = simple_texts
    file_model.images = []
    file_model.document_name = clean_filename
    file_model.file_id = file_id
    
    # 텍스트 청킹
    try:
        # 전체 텍스트 결합
        full_text = "\n".join(simple_texts)
        
        # utils의 split_text_into_chunks 사용
        from utils import split_text_into_chunks
        raw_chunks = split_text_into_chunks(full_text, None)  # tokenizer 없이 간단 청킹
        
        # 문서명 태깅
        file_model.chunks = [f"[문서: {clean_filename}] {chunk}" for chunk in raw_chunks]
        
        # 임베딩 생성
        file_model.embed_chunks()
        
        return file_model
        
    except Exception as e:
        # 폴백: 간단한 청킹
        chunk_size = 1000
        full_text = "\n".join(simple_texts)
        raw_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        file_model.chunks = [f"[문서: {clean_filename}] {chunk}" for chunk in raw_chunks]
        file_model.embed_chunks()
        return file_model

@app.post("/api/v1/files/")
async def upload_file_standard(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """표준 파일 업로드 엔드포인트"""
    return await _upload_file_logic(file)

@app.post("/v1/files/")
async def upload_file_v1(file: UploadFile = File(...)):
    """v1 파일 업로드 엔드포인트"""
    return await _upload_file_logic(file)

@app.put("/v1/process")
async def process_document_put_v1(request: Request):
    """OpenWebUI PUT 요청 문서 처리 엔드포인트"""
    try:
        # raw binary data 읽기
        file_data = await request.body()
        
        # 파일명 추출
        filename = "document.pdf"
        
        content_disposition = request.headers.get("content-disposition", "")
        if content_disposition:
            import re
            filename_match = re.search(r'filename[*]?=([^;\r\n"]*)', content_disposition)
            if filename_match:
                filename = filename_match.group(1).strip('"\'')
        elif request.headers.get("x-filename"):
            filename = request.headers.get("x-filename")
        elif request.query_params.get("filename"):
            filename = request.query_params.get("filename")
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"document_{timestamp}.pdf"
        
        # 파일명 정리
        import re
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        # UploadFile 객체 시뮬레이션
        class MockUploadFile:
            def __init__(self, file_path, content_type, filename):
                self.filename = filename
                self.content_type = content_type
                self._file_path = file_path
                
            async def read(self):
                with open(self._file_path, 'rb') as f:
                    return f.read()
        
        mock_file = MockUploadFile(temp_file_path, "application/pdf", filename)
        result = await _upload_file_logic(mock_file)
        
        # 분석 요약 생성
        file_id = result.id
        analysis_summary = f"""문서 처리 완료:
- 총 페이지: {result.data.get('total_pages', 0)}개
- 텍스트 청크: {result.data.get('text_chunks', 0)}개
- 처리 상태: {result.data.get('processing_status', 'completed')}

문서가 성공적으로 처리되었습니다. 문서 내용에 대해 질문해주세요!"""
        
        return {
            "page_content": analysis_summary,
            "metadata": {
                "filename": result.filename,
                "size": result.meta["size"],
                "content_type": result.meta["content_type"],
                "created_at": result.created_at,
                "file_id": result.id,
                "total_pages": result.data.get('total_pages', 0),
                "total_chunks": result.data.get('total_chunks', 0),
                "processing_status": result.data.get('processing_status', 'completed')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

async def _upload_file_logic(file: UploadFile):
    """파일 업로드 공통 로직"""
    global file_models, uploaded_files
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        # 고유 파일 ID 생성
        file_id = str(uuid.uuid4())
        
        # 임시 파일에 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # PDF 내용 추출
        pdf, simple_texts, simple_images = await extract_pdf_content(temp_file_path)
        
        # 문서명 설정
        clean_filename = file.filename.replace('.pdf', '').replace('.PDF', '')
        
        # 파일별 모델 생성 및 텍스트 청킹
        file_model = await create_file_model(file_id, pdf, simple_texts, simple_images, clean_filename)
        
        # 파일별 모델 저장
        file_models[file_id] = file_model
        
        # 파일 정보 저장
        text_chunks_count = len(file_model.chunks)
        file_info = {
            "id": file_id,
            "filename": file.filename,
            "document_name": clean_filename,
            "size": len(content),
            "content_type": file.content_type or "application/pdf",
            "model": "vqa_rag_client",
            "pages": len(pdf),
            "text_chunks": text_chunks_count,
            "image_chunks": 0,
            "total_chunks": text_chunks_count,
            "processed_at": datetime.now().isoformat(),
            "temp_file_path": temp_file_path,
            "image_processing_status": "pending"
        }
        
        uploaded_files[file_id] = file_info
        
        # 이미지 백그라운드 처리
        if simple_images:
            async def process_images_background():
                """백그라운드에서 이미지 배치 처리"""
                try:
                    # 이미지 설정
                    processed_images = []
                    for img_tuple in simple_images:
                        if isinstance(img_tuple, tuple) and len(img_tuple) >= 3:
                            page_num, img_num, pil_img = img_tuple
                            if hasattr(pil_img, 'format'):
                                processed_images.append(pil_img)
                    
                    # 파일 모델에 이미지 추가
                    if file_id in file_models:
                        file_models[file_id].images = processed_images
                        
                        # 배치 단위 이미지 처리
                        if processed_images:
                            # 파일 정보에 진행 상태 추가
                            if file_id in uploaded_files:
                                uploaded_files[file_id]["image_processing_status"] = "processing"
                                uploaded_files[file_id]["image_progress"] = 0
                                uploaded_files[file_id]["total_images"] = len(processed_images)
                            
                            # 진행 상태 추적을 위한 콜백 함수
                            async def progress_callback(batch_num, total_batches, processed_count, total_count):
                                progress = int((processed_count / total_count) * 100)
                                if file_id in uploaded_files:
                                    uploaded_files[file_id]["image_progress"] = progress
                                    uploaded_files[file_id]["processed_images"] = processed_count
                                print(f"📊 이미지 처리 진행률: {progress}% ({processed_count}/{total_count})")
                            
                            # 배치 처리 실행 (배치 크기 2)
                            await file_models[file_id].process_images_async(batch_size=2)
                            
                            # 파일 정보 업데이트
                            if file_id in uploaded_files:
                                image_chunks_count = len(getattr(file_models[file_id], 'image_chunks', []))
                                uploaded_files[file_id]["image_chunks"] = image_chunks_count
                                uploaded_files[file_id]["total_chunks"] = text_chunks_count + image_chunks_count
                                uploaded_files[file_id]["image_processing_status"] = "completed"
                                uploaded_files[file_id]["image_progress"] = 100
                                uploaded_files[file_id]["processed_images"] = len(processed_images)
                                
                                print(f"✅ 백그라운드: 이미지 처리 완료 - {image_chunks_count}개 이미지 청크")
                        else:
                            if file_id in uploaded_files:
                                uploaded_files[file_id]["image_processing_status"] = "no_images"
                    
                except Exception as e:
                    if file_id in uploaded_files:
                        uploaded_files[file_id]["image_processing_status"] = "error"
                        uploaded_files[file_id]["image_error"] = str(e)
            
            # 백그라운드 태스크 생성
            asyncio.create_task(process_images_background())
        else:
            # 이미지가 없는 경우
            if file_id in uploaded_files:
                uploaded_files[file_id]["image_processing_status"] = "no_images"
                uploaded_files[file_id]["image_progress"] = 100
        
        # OpenWebUI 표준 형식으로 응답
        current_time = int(time.time())
        
        return FileUploadResponse(
            id=file_id,
            user_id="system",
            hash=None,
            filename=file.filename,
            data={
                "total_pages": len(pdf),
                "total_chunks": text_chunks_count,
                "text_chunks": text_chunks_count,
                "image_chunks": 0,
                "processing_status": "text_completed",
                "image_processing_status": "pending",
                "document_name": clean_filename
            },
            meta={
                "name": file.filename,
                "content_type": file.content_type or "application/pdf",
                "size": len(content),
                "data": {
                    "total_pages": len(pdf),
                    "total_chunks": text_chunks_count,
                    "text_chunks": text_chunks_count,
                    "image_chunks": 0,
                    "processing_status": "text_completed",
                    "image_processing_status": "pending",
                    "document_name": clean_filename
                }
            },
            created_at=current_time,
            updated_at=current_time
        )
        
    except Exception as e:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenWebUI 호환 채팅 완료 API"""
    return await _chat_completions_logic(request)

@app.post("/api/chat/completions")
async def chat_completions_api(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """OpenWebUI 표준 채팅 완료 API"""
    return await _chat_completions_logic(request)

@app.post("/api/v1/chat/completions")
async def chat_completions_v1(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """OpenWebUI v1 채팅 완료 API"""
    return await _chat_completions_logic(request)

async def _chat_completions_logic(request: ChatRequest):
    """채팅 완료 공통 로직 - 모델 서버 클라이언트 사용"""
    global file_models, uploaded_files, model_client
    
    # 이미지 입력 bytes로 처리
    user_message = None
    image_bytes_list = []
    image_urls_from_content = []
    
    for msg in reversed(request.messages):
        content = msg.content
        # content가 리스트(복합 메시지)인 경우
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif item.get("type") == "image_url" and "image_url" in item:
                        image_url = item["image_url"]
                        if isinstance(image_url, dict) and "url" in image_url:
                            url = image_url["url"]
                            if url.startswith("data:image/"):
                                import re, base64
                                match = re.match(r"data:image/[^;]+;base64,(.*)", url)
                                if match:
                                    img_bytes = base64.b64decode(match.group(1))
                                    image_bytes_list.append(img_bytes)
                            else:
                                image_urls_from_content.append(url)
                        elif isinstance(image_url, str):
                            if image_url.startswith("data:image/"):
                                import re, base64
                                match = re.match(r"data:image/[^;]+;base64,(.*)", image_url)
                                if match:
                                    img_bytes = base64.b64decode(match.group(1))
                                    image_bytes_list.append(img_bytes)
                            else:
                                image_urls_from_content.append(image_url)
            if texts:
                user_message = "\n".join(texts)
            break
        elif isinstance(content, str) and content.strip():
            user_message = content.strip()
            break
    
    # request.images도 bytes로 처리
    if hasattr(request, 'images') and request.images:
        for img in request.images:
            if isinstance(img, bytes):
                image_bytes_list.append(img)
            elif isinstance(img, str):
                import base64
                try:
                    image_bytes_list.append(base64.b64decode(img))
                except Exception:
                    pass
            elif isinstance(img, dict) and 'data' in img:
                import base64
                try:
                    image_bytes_list.append(base64.b64decode(img['data']))
                except Exception:
                    pass
    
    # image_urls도 처리
    if hasattr(request, 'image_urls') and request.image_urls:
        for img_url_dict in request.image_urls:
            url = img_url_dict.get('url')
            if url:
                image_urls_from_content.append(url)
    
    # 이미지만 있을 때 VQA
    if not uploaded_files and image_bytes_list:
        try:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes_list[0]))
            prompt = user_message if user_message else "이 이미지는 무엇인가요?"
            answer = model_client.generate_vqa(prompt, img)
        except Exception as e:
            answer = "이미지 분석 중 오류가 발생했습니다."
        
        response = ChatResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()) if prompt else 0,
                "completion_tokens": len(answer.split()) if answer else 0,
                "total_tokens": (len(prompt.split()) if prompt else 0) + (len(answer.split()) if answer else 0)
            }
        )
        
        if request.stream:
            return create_streaming_response(answer, request.model)
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content=response.dict())
    
    # 텍스트+이미지 혼합 시 이미지 요약
    image_summaries = []
    if image_bytes_list:
        from PIL import Image
        from io import BytesIO
        file_model = next(iter(file_models.values())) if file_models else None
        for idx, img_bytes in enumerate(image_bytes_list):
            try:
                img = Image.open(BytesIO(img_bytes))
                if file_model:
                    summary = file_model.summarize_image(img)
                    if summary:
                        image_summaries.append(f"🖼️ 첨부 이미지 {idx+1}: {summary}")
            except Exception as e:
                pass
    
    # image_urls 처리
    if image_urls_from_content:
        from PIL import Image
        from io import BytesIO
        file_model = next(iter(file_models.values())) if file_models else None
        for idx, url in enumerate(image_urls_from_content):
            try:
                resp = req.get(url)
                img = Image.open(BytesIO(resp.content))
                if file_model:
                    summary = file_model.summarize_image(img)
                    if summary:
                        image_summaries.append(f"🖼️ 첨부 이미지 URL {idx+1}: {summary}")
            except Exception as e:
                pass
    
    # 모델 서버 연결 확인
    if not model_client or not model_client.is_healthy():
        raise HTTPException(status_code=500, detail="모델 서버에 연결할 수 없습니다. 모델 서버를 시작해주세요.")
    
    try:
        # system_message도 추출
        system_message = None
        for msg in request.messages:
            if hasattr(msg, "role") and msg.role == "system" and msg.content:
                system_message = msg.content
                break
        
        # OpenWebUI 시스템 프롬프트 처리
        if not user_message:
            for msg in request.messages:
                if msg.content and any(keyword in msg.content for keyword in ["### Task:", "Generate a concise", "title", "JSON format"]):
                    if "search queries" in msg.content.lower():
                        answer = "No search queries needed."
                    elif "title" in msg.content.lower():
                        if uploaded_files:
                            answer = f'{{"title": "📄 문서 분석"}}'
                        else:
                            answer = '{"title": "💬 일반 대화"}'
                    elif "tags" in msg.content.lower():
                        if uploaded_files:
                            answer = '{"tags": ["Document", "PDF", "Analysis", "RAG"]}'
                        else:
                            answer = '{"tags": ["General", "Conversation"]}'
                    else:
                        answer = "Task completed."
                    
                    if request.stream:
                        return create_streaming_response(answer, request.model)
                    
                    response = ChatResponse(
                        id=f"chatcmpl-{str(uuid.uuid4())}",
                        object="chat.completion",
                        created=int(datetime.now().timestamp()),
                        model=request.model,
                        choices=[{
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop"
                        }],
                        usage={
                            "prompt_tokens": len(msg.content.split()) if msg.content else 0,
                            "completion_tokens": len(answer.split()) if answer else 0,
                            "total_tokens": (len(msg.content.split()) if msg.content else 0) + (len(answer.split()) if answer else 0)
                        }
                    )
                    from fastapi.responses import JSONResponse
                    return JSONResponse(content=response.dict())
            else:
                # 환영 메시지
                if uploaded_files:
                    file_list = []
                    for file_id, file_info in uploaded_files.items():
                        file_list.append(f"📄 **문서** ({file_info['total_chunks']}개 청크)")
                    
                    answer = f"""안녕하세요! 현재 업로드된 문서가 있습니다:

{chr(10).join(file_list)}

문서에 대해 질문해주세요! 예를 들어:
- "이 문서의 주요 내용은 무엇인가요?"
- "중요한 정보를 요약해주세요" """
                else:
                    answer = "안녕하세요! 무엇을 도와드릴까요? PDF 문서를 업로드하시면 문서 내용에 대해 질문하실 수 있습니다."
        else:
            # 디버깅 암호문 체크
            if detect_debug_command(user_message):
                answer = print_chunk_debug(file_models, uploaded_files)
            
            # 시스템 메시지에서 RAG 컨텍스트 추출
            elif system_message and "<context>" in system_message:
                import re
                context_match = re.search(r'<context>(.*?)</context>', system_message, re.DOTALL)
                if context_match:
                    rag_context = context_match.group(1).strip()
                    
                    # 의미없는 컨텍스트 체크
                    useless_keywords = [
                        "문서 처리 완료",
                        "문서가 성공적으로 처리되었습니다",
                        "처리 상태:",
                        "텍스트 청크:",
                        "총 페이지:",
                        "문서 내용에 대해 질문해주세요"
                    ]
                    
                    is_useless_context = any(keyword in rag_context for keyword in useless_keywords)
                    
                    if is_useless_context:
                        answer = None  # 파일 기반 RAG로 전환
                    else:
                        rag_prompt = f"""다음 컨텍스트를 바탕으로 사용자의 질문에 답변해주세요.

컨텍스트:
{rag_context}

질문: {user_message}

답변:"""
                        answer = model_client.generate_text(rag_prompt)
                else:
                    answer = "죄송합니다. 컨텍스트를 찾을 수 없습니다."
            
            # 업로드된 파일이 있으면 파일 기반 RAG 사용
            if uploaded_files and (answer is None or not answer):
                if len(file_models) > 1:
                    # 다중 문서 모드
                    best_file_id = None
                    best_score = -float('inf')
                    for file_id, file_model in file_models.items():
                        results = file_model.retrieve(user_message, top_k=10)
                        score_sum = sum([score for _, _, score in results])
                        if score_sum > best_score:
                            best_score = score_sum
                            best_file_id = file_id
                    
                    if best_file_id is not None:
                        file_model = file_models[best_file_id]
                        
                        # 이미지 요약도 컨텍스트에 추가
                        if image_summaries:
                            file_model.chunks = image_summaries + file_model.chunks
                        
                        answer = file_model.QnA(user_message, top_k=5)
                        answer = f"📚 **다중 문서 검색** (문서: {file_model.document_name})\n\n{answer}\n\n---\n📊 **총 문서**: {len(file_models)}개"
                    else:
                        answer = "죄송합니다. 관련 문서를 찾을 수 없습니다."
                else:
                    # 단일 문서 모드
                    file_id = list(file_models.keys())[0]
                    file_model = file_models[file_id]
                    file_info = uploaded_files[file_id]
                    
                    if file_model.chunks:
                        # 이미지 요약도 컨텍스트에 추가
                        if image_summaries:
                            file_model.chunks = image_summaries + file_model.chunks
                        
                        answer = file_model.QnA(user_message, top_k=5)
                        answer = f"📄 문서 기반 답변:\n\n{answer}\n\n---\n📊 **문서 정보**: {file_info['pages']}페이지, {file_info['total_chunks']}개 청크"
                    else:
                        answer = "죄송합니다. 파일이 아직 처리되지 않았습니다."
            else:
                # 일반 대화 모드
                if not uploaded_files:
                    prompt = f"질문: {user_message}\n답변을 해주세요."
                    answer = model_client.generate_text(prompt)
                else:
                    answer = "죄송합니다. 업로드된 문서가 아직 처리 중이거나 문제가 있습니다."
        
        # 답변 검증
        if not answer or not answer.strip():
            answer = "죄송합니다. 질문을 이해하지 못했습니다. 다른 방식으로 질문해주세요."
        
        # 스트리밍 요청 처리
        if request.stream:
            return create_streaming_response(answer, request.model)
        
        # 일반 응답
        response = ChatResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_message.split()) if user_message else 0,
                "completion_tokens": len(answer.split()) if answer else 0,
                "total_tokens": (len(user_message.split()) if user_message else 0) + (len(answer.split()) if answer else 0)
            }
        )
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content=response.dict())
        
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류: {str(e)}")

def create_streaming_response(answer: str, request_model: str):
    """스트리밍 응답 생성"""
    async def generate():
        chunk_id = f"chatcmpl-{str(uuid.uuid4())}"
        created = int(datetime.now().timestamp())
        
        # 첫 번째 청크
        first_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
        
        # 답변을 청크로 나누어 전송
        words = answer.split()
        chunk_size = 3
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            content = " ".join(chunk_words)
            if i > 0:
                content = " " + content
                
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)
        
        # 마지막 청크
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/v1/status")
async def get_status():
    """현재 상태 확인 - 이미지 처리 진행 상황 포함"""
    model_server_status = "disconnected"
    if model_client:
        model_server_status = "connected" if model_client.is_healthy() else "unhealthy"
    
    # 파일별 상세 정보 구성
    files_info = {}
    for file_id, info in uploaded_files.items():
        file_detail = {
            "filename": info["filename"],
            "size": info["size"],
            "total_chunks": info["total_chunks"],
            "text_chunks": info.get("text_chunks", 0),
            "image_chunks": info.get("image_chunks", 0),
            "processed_at": info["processed_at"],
            "has_model": file_id in file_models,
            "image_processing_status": info.get("image_processing_status", "unknown"),
            "image_progress": info.get("image_progress", 0),
            "total_images": info.get("total_images", 0),
            "processed_images": info.get("processed_images", 0)
        }
        
        if "image_error" in info:
            file_detail["image_error"] = info["image_error"]
        
        files_info[file_id] = file_detail
    
    return {
        "model_server_status": model_server_status,
        "uploaded_files": len(uploaded_files),
        "file_models": len(file_models),
        "files_info": files_info,
        "processing_summary": {
            "total_files": len(uploaded_files),
            "completed_text_processing": len([f for f in uploaded_files.values() if f.get("text_chunks", 0) > 0]),
            "completed_image_processing": len([f for f in uploaded_files.values() if f.get("image_processing_status") == "completed"]),
            "processing_images": len([f for f in uploaded_files.values() if f.get("image_processing_status") == "processing"]),
            "total_text_chunks": sum(f.get("text_chunks", 0) for f in uploaded_files.values()),
            "total_image_chunks": sum(f.get("image_chunks", 0) for f in uploaded_files.values()),
            "total_chunks": sum(f.get("total_chunks", 0) for f in uploaded_files.values())
        }
    }

@app.delete("/api/v1/files/{file_id}")
async def delete_file(file_id: str):
    """파일 삭제"""
    global file_models, uploaded_files
    
    try:
        if file_id in uploaded_files:
            file_info = uploaded_files[file_id]
            
            if 'temp_file_path' in file_info and os.path.exists(file_info['temp_file_path']):
                os.unlink(file_info['temp_file_path'])
            
            del uploaded_files[file_id]
        
        if file_id in file_models:
            del file_models[file_id]
        
        return {
            "message": f"파일 {file_id}가 성공적으로 삭제되었습니다.",
            "deleted_file_id": file_id,
            "remaining_files": len(uploaded_files),
            "remaining_models": len(file_models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류: {str(e)}")

@app.delete("/v1/files/{file_id}")
async def delete_file_v1(file_id: str):
    """OpenAI 호환 파일 삭제 엔드포인트"""
    return await delete_file(file_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
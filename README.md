# VQA RAG System

OpenWebUI 호환 **PDF 문서 분석 및 RAG** 시스템입니다. 
**Visual Question Answering (VQA)** 기능을 포함하여 텍스트와 이미지가 포함된 PDF 문서를 분석하고 질문에 답변할 수 있습니다.

## ✨ 주요 특징

- 🔗 **OpenWebUI 완전 호환**: 표준 OpenAI API 형식 지원
- 📄 **다중 PDF 처리**: 여러 문서 동시 분석 및 검색
- 🖼️ **이미지 + 텍스트 혼합**: VQA 모델로 이미지 내용 분석
- ⚡ **실시간 스트리밍**: 점진적 답변 생성
- 🧠 **KV 캐시 최적화**: 반복 질문 시 30-50% 속도 향상
- 🔄 **백그라운드 처리**: 이미지 분석을 백그라운드에서 배치 처리
- 🛠️ **디버깅 지원**: 청크 분석 및 상태 확인 도구

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │───▶│   API Server    │───▶│  Model Server   │
│  (Frontend)     │    │  (Port: 8000)   │    │  (Port: 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                      ┌─────────────────┐    ┌─────────────────┐
                      │  Models Client  │    │ GPU Models      │
                      │  (Wrapper)      │    │ - Text LLM      │
                      └─────────────────┘    │ - VQA Model     │
                              │              │ - Embedder      │
                              ▼              └─────────────────┘
                      ┌─────────────────┐
                      │   Utilities     │
                      │ - PDF Parser    │
                      │ - Text Chunker  │
                      │ - Debug Tools   │
                      └─────────────────┘
```

## 🚀 빠른 시작

### 1. 저장소 클론 및 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/vqa-rag-system.git
cd vqa-rag-system

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드 (최초 1회)
python -c "import nltk; nltk.download('punkt')"
```

### 2. GPU 설정 확인

```bash
# CUDA 사용 가능 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# GPU 메모리 확인 (8GB 이상 권장)
nvidia-smi
```

### 3. 서버 시작

**🔥 중요: 반드시 순서대로 시작해주세요!**

```bash
# Step 1: 모델 서버 시작 (GPU 필요, 첫 실행 시 모델 다운로드로 시간 소요)
python model_server.py

# Step 2: 새 터미널에서 API 서버 시작
python api_server.py
```

### 4. OpenWebUI 연결

1. **OpenWebUI 설치** (별도 설치 필요):
   ```bash
   pip install open-webui
   # 또는 Docker: docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway ghcr.io/open-webui/open-webui:main
   ```

2. **모델 추가**:
   - OpenWebUI 설정 → Models → Add Model
   - **Base URL**: `http://localhost:8000`
   - **API Key**: `test-api-key` (또는 `your-api-key-here`)
   - **Model ID**: `vqa_rag` (또는 `text_only_rag`, `async_vqa_rag_cag`)

## 📖 사용법

### 📄 PDF 문서 업로드
1. OpenWebUI에서 PDF 파일 드래그 앤 드롭
2. 자동으로 텍스트 추출 및 청킹 처리
3. 이미지 분석은 백그라운드에서 진행

### 💬 질문하기
```
이 문서의 주요 내용은 무엇인가요?
3페이지에서 언급된 가격 정보를 찾아주세요
이미지에 나온 차트를 분석해주세요
```

### 🔍 디버깅
```
청크 출력     # 처리된 청크들 확인
print chunk   # 영어 명령어
```

## 📁 파일 구조

```
vqa-rag-system/
├── 📄 README.md              # 프로젝트 가이드
├── 📄 requirements.txt       # 패키지 의존성
├── 🌐 api_server.py          # OpenWebUI 호환 API 서버 (Port 8000)
├── 🧠 model_server.py        # GPU 모델 추론 서버 (Port 8001)
├── 🔗 models_client.py       # 모델 서버 클라이언트
├── 🛠️ utils.py               # 핵심 유틸리티 (PDF 파싱, 텍스트 청킹)
└── 🐛 debug_chunk.py         # 청크 디버깅 도구
```

### 파일별 상세 역할

| 파일 | 역할 | 주요 기능 |
|------|------|-----------|
| `api_server.py` | OpenWebUI 호환 API 서버 | PDF 업로드, 채팅 API, 스트리밍 응답 |
| `model_server.py` | GPU 모델 추론 서버 | 텍스트 생성, VQA, 임베딩 생성 |
| `models_client.py` | 모델 서버 클라이언트 | HTTP 통신, 배치 처리, RAG 구현 |
| `utils.py` | 핵심 유틸리티 | PDF 파싱, 텍스트 청킹, AI 인터페이스 |
| `debug_chunk.py` | 디버깅 도구 | 청크 분석, 상태 확인 |

## 🤖 지원 모델

### 텍스트 생성
- **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B**
- GPU 1에 할당, 4bit 양자화

### VQA (Visual Question Answering)  
- **naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B**
- GPU 3에 할당, FP16

### 임베딩
- **jhgan/ko-sroberta-multitask**
- 한국어 특화 문장 임베딩

## 📊 API 엔드포인트

### API 서버 (Port 8000)
```
POST /v1/chat/completions      # 채팅 완료 (OpenAI 호환)
POST /v1/files/                # 파일 업로드
PUT  /v1/process              # 문서 처리
GET  /v1/status               # 시스템 상태
GET  /v1/models               # 모델 목록
DEL  /v1/files/{file_id}      # 파일 삭제
```

### 모델 서버 (Port 8001)
```
POST /embed                   # 텍스트 임베딩
POST /text/generate           # 텍스트 생성
POST /vqa/generate            # VQA 생성
GET  /health                  # 모델 상태
```

## ⚙️ 설정 및 최적화

### GPU 메모리 할당
```python
# model_server.py에서 설정
GPU 1: Text 모델 (약 3GB)
GPU 3: VQA 모델 (약 6GB)
GPU 0: 임베딩 모델 (약 1GB, 자동 할당)
```

### 환경 변수
```bash
export CUDA_VISIBLE_DEVICES=0,1,3  # 사용할 GPU 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 메모리 최적화
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 모델 서버 연결 실패
```bash
# 확인 사항
- model_server.py가 먼저 실행되었는지 확인
- GPU 메모리 부족 확인 (nvidia-smi)
- 포트 8001이 사용 중인지 확인 (netstat -tulpn | grep 8001)
```

#### 2. PDF 처리 실패
```bash
# 확인 사항  
- 파일 크기 제한 (100MB 이하 권장)
- PDF 형식 확인 (스캔된 PDF는 OCR 필요)
- 임시 파일 디렉토리 권한 확인
```

#### 3. 이미지 분석 느림
```bash
# 해결 방법
- GPU 메모리 확인
- 배치 크기 조정 (models_client.py에서 batch_size)
- 백그라운드 처리 상태 확인 (/v1/status 엔드포인트)
```

### 성능 최적화

#### KV 캐시 활용
- 자주 검색되는 청크들을 자동으로 캐시
- 반복 질문 시 30-50% 속도 향상
- 5번의 검색마다 캐시 자동 업데이트

#### 배치 처리
- 이미지 분석을 배치 단위로 처리 (기본: 2개씩)
- 메모리 사용량과 속도의 균형 최적화

## 📋 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU 8GB+ (RTX 3070, A100, V100 등)
- **RAM**: 16GB+ 권장
- **Storage**: 10GB+ (모델 다운로드 포함)

### 소프트웨어
- **Python**: 3.8+
- **CUDA**: 11.8+ 또는 12.0+
- **PyTorch**: 2.0+

### 브라우저 지원
- **Chrome, Firefox, Safari**: 최신 버전
- **WebSocket**: 스트리밍 응답용

## 📈 성능 벤치마크

| 작업 | 처리 시간 | 메모리 사용량 |
|------|-----------|---------------|
| PDF 업로드 (10페이지) | ~5초 | ~2GB |
| 텍스트 청킹 (50개) | ~2초 | ~500MB |
| 이미지 분석 (5개) | ~10초 | ~4GB |
| 질문 답변 (캐시 없음) | ~3초 | ~1GB |
| 질문 답변 (캐시 있음) | ~1.5초 | ~800MB |

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📜 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 

---
### 🏷️ Keywords
`python` `fastapi` `rag` `vqa` `openwebui` `pdf` `ai` `pytorch` `transformers` `korean-nlp`
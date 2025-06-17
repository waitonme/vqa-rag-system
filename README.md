# VQA RAG System

OpenWebUI 호환 PDF 문서 분석 및 RAG (Retrieval-Augmented Generation) 시스템입니다. 
Visual Question Answering (VQA) 기능을 포함하여 텍스트와 이미지가 포함된 PDF 문서를 분석하고 질문에 답변할 수 있습니다.

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

## 📁 파일별 역할

### 🌐 API 서버 계층

#### `api_server.py` - 메인 API 서버
- **역할**: OpenWebUI 호환 RESTful API 서버
- **기능**:
  - PDF 파일 업로드 및 처리 (`/v1/files/`, `/v1/process`)
  - 채팅 완료 API (`/v1/chat/completions`)
  - 스트리밍 응답 지원
  - 다중 문서 RAG 처리
  - 이미지 + 텍스트 혼합 질의응답
- **포트**: 8000
- **의존성**: `models_client.py`, `utils.py`, `debug_chunk.py`

#### `model_server.py` - GPU 모델 추론 서버
- **역할**: GPU에서 실행되는 AI 모델들의 추론 서버
- **기능**:
  - 텍스트 임베딩 생성 (`/embed`)
  - 텍스트 생성 (`/text/generate`)
  - VQA (Visual Question Answering) (`/vqa/generate`)
  - 모델 상태 확인 (`/health`)
- **포트**: 8001
- **로드 모델**: 
  - `jhgan/ko-sroberta-multitask` (임베딩)
  - `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` (텍스트 생성)
  - `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` (VQA)

### 🔗 클라이언트 계층

#### `models_client.py` - 모델 서버 클라이언트
- **역할**: API 서버와 모델 서버 간의 통신 인터페이스
- **주요 클래스**:
  - `ModelServerClient`: HTTP 통신 클라이언트
  - `VQARAGModelWrapper`: 기존 인터페이스 호환성 유지
- **기능**:
  - 모델 서버 API 호출
  - 이미지 배치 처리
  - 임베딩 기반 유사도 검색
  - 문서별 RAG 처리

### 🛠️ 유틸리티 계층

#### `utils.py` - 핵심 유틸리티 함수들
- **역할**: 텍스트 처리, PDF 파싱, AI 모델 인터페이스
- **주요 함수**:
  - `split_text_into_chunks()`: 토큰 기반 텍스트 청킹
  - `simple_parse()`: PDF에서 텍스트와 이미지 추출
  - `hf_generate_response()`: HuggingFace 모델 텍스트 생성
  - `vqa_generate_response()`: VQA 모델 이미지 분석
  - `clean_text()`: 텍스트 정리 및 전처리
  - `extract_assistant_response()`: 모델 응답 파싱

#### `debug_chunk.py` - 디버깅 도구
- **역할**: 청크 분석 및 디버깅 지원
- **기능**:
  - `print_chunk_debug()`: 전체 청크 내용 출력
  - `detect_debug_command()`: 디버깅 명령어 감지
- **사용법**: 채팅에서 "청크 출력" 또는 "print chunk" 입력

#### `visualization.py` - 성능 분석 및 시각화
- **역할**: 모델 평가 결과 시각화 및 분석
- **기능**:
  - 모델별 성능 비교 차트
  - PDF 파일별 성능 분석
  - 카테고리별 성능 분석
  - 종합 요약 보고서 생성
- **출력**: PNG 차트, 텍스트 요약

## 🚀 시작하기

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드 (최초 1회)
python -c "import nltk; nltk.download('punkt')"
```

### 2. 서버 시작

**Step 1: 모델 서버 시작 (GPU 필요)**
```bash
python model_server.py
```

**Step 2: API 서버 시작**
```bash
python api_server.py
```

### 3. OpenWebUI 연결
- OpenWebUI에서 새 모델 추가
- Base URL: `http://localhost:8000`
- API Key: `test-api-key` 또는 `your-api-key-here`

## 🔧 사용법

### 1. PDF 문서 업로드
- OpenWebUI에서 PDF 파일을 업로드
- 자동으로 텍스트 추출 및 청킹
- 이미지 분석 (백그라운드 처리)

### 2. 질문하기
```
이 문서의 주요 내용은 무엇인가요?
특정 정보를 찾아주세요
이미지에 대해 설명해주세요
```

### 3. 디버깅
```
청크 출력    # 처리된 청크들을 확인
print chunk  # 영어 명령어
```

## 📊 지원하는 모델들

| 모델 ID | 설명 |
|---------|------|
| `text_only_rag` | 텍스트만 처리하는 RAG |
| `vqa_rag` | 텍스트 + 이미지 VQA RAG |
| `vqa_rag_cag` | 개선된 VQA RAG |
| `async_vqa_rag_cag` | 비동기 처리 VQA RAG |

## 🎯 주요 특징

- **OpenWebUI 완전 호환**: 표준 OpenAI API 형식 지원
- **다중 문서 처리**: 여러 PDF 동시 분석
- **실시간 이미지 분석**: 백그라운드 배치 처리
- **스트리밍 응답**: 실시간 답변 생성
- **디버깅 지원**: 청크 분석 및 상태 확인
- **GPU 분리**: 모델 서버 별도 운영으로 확장성 확보

## 🔍 API 엔드포인트

### API 서버 (Port 8000)
- `POST /v1/chat/completions` - 채팅 완료
- `POST /v1/files/` - 파일 업로드
- `PUT /v1/process` - 문서 처리
- `GET /v1/status` - 시스템 상태
- `GET /v1/models` - 모델 목록

### 모델 서버 (Port 8001)
- `POST /embed` - 텍스트 임베딩
- `POST /text/generate` - 텍스트 생성
- `POST /vqa/generate` - VQA 생성
- `GET /health` - 모델 상태

## 🔧 설정

### GPU 메모리 할당
- GPU 1: Text 모델 (1.5B 파라미터)
- GPU 3: VQA 모델 (3B 파라미터)
- GPU 0: 임베딩 모델 (자동 할당)

### 환경 변수
```bash
export CUDA_VISIBLE_DEVICES=0,1,3  # 사용할 GPU 설정
```

## 📈 성능 모니터링

`visualization.py`를 사용하여 성능 분석:
```bash
python visualization.py
```

생성되는 분석 차트:
- 모델별 성능 비교
- 파일별 처리 성능
- 카테고리별 분석
- 종합 요약 보고서

## 🐛 문제 해결

### 일반적인 문제들
1. **모델 서버 연결 실패**: 
   - `model_server.py`가 먼저 실행되었는지 확인
   - GPU 메모리 부족 확인

2. **PDF 처리 실패**:
   - 파일 크기 및 형식 확인
   - 임시 파일 경로 권한 확인

3. **이미지 분석 느림**:
   - 백그라운드 처리 진행 상황 확인
   - GPU 메모리 사용량 확인

### 로그 확인
- API 서버: 요청별 처리 시간 로깅
- 모델 서버: 모델 로딩 및 추론 상태
- 시스템 상태: `/v1/status` 엔드포인트 활용

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 
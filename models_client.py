import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Any
import json
import asyncio

class ModelServerClient:
    """모델 서버와 통신하는 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """모델 서버 상태 확인"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def embed_texts(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """텍스트 임베딩 생성"""
        data = {
            "texts": texts,
            "convert_to_numpy": convert_to_numpy
        }
        
        response = self.session.post(f"{self.base_url}/embed", json=data)
        response.raise_for_status()
        
        result = response.json()
        embeddings = result["embeddings"]
        
        if convert_to_numpy:
            return np.array(embeddings)
        else:
            return embeddings
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     system: Optional[str] = None, kv_cache: Optional[Dict] = None) -> str:
        """텍스트 생성"""
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "system": system,
            "kv_cache": kv_cache
        }
        
        response = self.session.post(f"{self.base_url}/text/generate", json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    
    def generate_vqa(self, prompt: str, image: Image.Image, max_new_tokens: int = 256) -> str:
        """VQA 생성"""
        # 이미지를 base64로 인코딩
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        data = {
            "prompt": prompt,
            "image_base64": image_base64,
            "max_new_tokens": max_new_tokens
        }
        
        response = self.session.post(f"{self.base_url}/vqa/generate", json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    
    def list_models(self) -> Dict[str, Any]:
        """로드된 모델 목록 조회"""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def is_healthy(self) -> bool:
        """모델 서버가 정상 동작하는지 확인"""
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except:
            return False

class VQARAGModelWrapper:
    """기존 VQA RAG 모델 인터페이스를 유지하면서 모델 서버 사용"""
    
    def __init__(self, model_client: ModelServerClient):
        self.client = model_client
        self.chunks = []
        self.chunk_embeddings = []
        self.texts = []
        self.images = []
        self.image_chunks = []
        self.image_embeddings = []
        self.pdf = None
        
        # 캐시 관련
        self.kv_cache = None
        self.kv_cache_context = None  # 캐시된 컨텍스트 추가
        self.retrievals_count = []
        self.top_k_chunk_idx = []
        
        # 문서 정보
        self.document_name = "Unknown Document"
        self.file_id = None
    
    def build_kv_cache(self, top_k: int = 10):
        """자주 검색되는 청크들로 KV 캐시 생성"""
        if not self.chunks or len(self.chunk_embeddings) == 0:
            return
            
        try:
            # 가장 자주 검색되는 청크들 선택
            if len(self.retrievals_count) > 0:
                # 검색 빈도 기반 선택
                frequent_indices = sorted(range(len(self.retrievals_count)), 
                                        key=lambda i: self.retrievals_count[i], reverse=True)[:top_k]
                cache_chunks = [self.chunks[i] for i in frequent_indices if i < len(self.chunks)]
            else:
                # 처음에는 상위 청크들 선택
                cache_chunks = self.chunks[:min(top_k, len(self.chunks))]
            
            if cache_chunks:
                # 캐시용 컨텍스트 구성
                self.kv_cache_context = "\n".join(cache_chunks)
                
                # 캐시 생성 요청 (모델 서버에서 처리하도록 확장 필요)
                # 현재는 컨텍스트만 저장하고 generate_text에서 활용
                print(f"✅ KV 캐시 준비 완료: {len(cache_chunks)}개 청크")
                
        except Exception as e:
            print(f"❌ KV 캐시 생성 실패: {e}")
    
    def embed_chunks(self):
        """청크들을 임베딩으로 변환"""
        if not self.chunks:
            return
            
        try:
            self.chunk_embeddings = self.client.embed_texts(self.chunks, convert_to_numpy=True)
            # 임베딩 생성 후 KV 캐시 구축
            self.build_kv_cache()
        except Exception as e:
            self.chunk_embeddings = []
    
    async def process_images_async(self, batch_size: int = 3):
        """이미지 배치 단위 비동기 처리 - 실시간 청크 업데이트"""
        if not self.images:
            return
        
        # 배치 단위로 이미지 처리
        total_processed = 0
        batch_count = 0
        
        for i in range(0, len(self.images), batch_size):
            batch_count += 1
            batch_images = self.images[i:i+batch_size]
            
            # 현재 배치의 이미지 요약 생성
            batch_summaries = []
            for j, image in enumerate(batch_images):
                image_idx = i + j
                try:
                    summary = self.summarize_image(image)
                    if summary and summary.strip() and summary.strip() != "None":
                        tagged_summary = f"[이미지 {image_idx+1} - 문서: {self.document_name}] {summary}"
                        batch_summaries.append(tagged_summary)
                except Exception as e:
                    pass
            
            # 배치 요약이 있으면 즉시 업데이트
            if batch_summaries:
                # 이미지 청크에 추가
                self.image_chunks.extend(batch_summaries)
                
                # 배치 임베딩 생성
                try:
                    batch_embeddings = self.client.embed_texts(batch_summaries, convert_to_numpy=True)
                    
                    # 이미지 임베딩에 추가
                    if len(self.image_embeddings) == 0:
                        self.image_embeddings = batch_embeddings
                    else:
                        self.image_embeddings = np.vstack([self.image_embeddings, batch_embeddings])
                    
                    # 전체 청크에 추가
                    self.chunks.extend(batch_summaries)
                    
                    # 전체 임베딩 업데이트
                    if len(self.chunk_embeddings) == 0:
                        self.chunk_embeddings = self.image_embeddings.copy()
                    else:
                        text_chunk_count = len(self.chunks) - len(self.image_chunks)
                        if text_chunk_count > 0:
                            text_embeddings = self.chunk_embeddings[:text_chunk_count]
                            self.chunk_embeddings = np.vstack([text_embeddings, self.image_embeddings])
                        else:
                            self.chunk_embeddings = self.image_embeddings.copy()
                    
                except Exception as e:
                    pass
            
            total_processed += len(batch_images)
            
            # 배치 간 짧은 대기
            if i + batch_size < len(self.images):
                await asyncio.sleep(0.1)
    
    def retrieve(self, question: str, top_k: int = 4):
        """관련 청크 검색 (유사도 점수 포함)"""
        if len(self.chunk_embeddings) == 0:
            return []
        
        try:
            # 질문 임베딩
            q_emb = self.client.embed_texts([question], convert_to_numpy=True)[0]
            
            # 유사도 계산
            sims = np.dot(self.chunk_embeddings, q_emb)
            
            if len(sims) == 0:
                return []
            
            # Top-k 선택
            topk_idx = np.argsort(sims)[::-1][:top_k]
            return [(i, self.chunks[i], float(sims[i])) for i in topk_idx if i < len(self.chunks)]
            
        except Exception as e:
            return []
    
    def QnA(self, question: str, top_k: int = 8) -> str:
        """질문 답변"""
        relevant_chunks = self.retrieve(question, top_k=top_k)
        
        if not relevant_chunks:
            prompt = f"질문: {question}\n답변을 해주세요."
            return self.client.generate_text(prompt)
        
        # 검색 카운트 업데이트
        while len(self.retrievals_count) < len(self.chunks):
            self.retrievals_count.append(0)
        
        for i, _, _ in relevant_chunks:
            if i < len(self.retrievals_count):
                self.retrievals_count[i] += 1
        
        # KV 캐시 업데이트 (주기적으로)
        if len(self.retrievals_count) > 0 and sum(self.retrievals_count) % 5 == 0:
            self.build_kv_cache()
        
        # 컨텍스트 구성
        context = "\n".join([chunk for _, chunk, _ in relevant_chunks])
        
        # KV 캐시 활용한 프롬프트 구성
        if self.kv_cache_context:
            # 캐시된 컨텍스트와 새로운 컨텍스트 결합
            full_context = f"기본 문서 내용:\n{self.kv_cache_context}\n\n관련 내용:\n{context}"
            prompt = f"""당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요.
다음은 문서의 일부 내용입니다:
{full_context}

위의 정보를 바탕으로 다음 질문에 답변해주세요:

{question}"""
        else:
            # 일반적인 프롬프트
            prompt = f"""당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요.
다음은 문서의 일부 내용입니다:
{context}

위의 정보를 바탕으로 다음 질문에 답변해주세요:

{question}"""
        
        try:
            # KV 캐시가 있으면 전달 (향후 모델 서버에서 지원)
            # 현재는 컨텍스트 최적화로 성능 향상
            return self.client.generate_text(prompt)
        except Exception as e:
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def summarize_image(self, image: Image.Image) -> Optional[str]:
        """이미지 요약"""
        if isinstance(image, tuple):
            image = image[2]
        if image is None or not isinstance(image, Image.Image):
            return None
            
        prompt = """당신은 최고의 VQA 모델입니다.
다음 이미지는 PDF 에서 추출한 그림입니다.
1. VQA 처럼 한글로 요약해 주세요.
2. 만약 아이콘, 버튼, 로고, 구분선, 백그라운드 이미지 라고 이라고 생각되면 **None**만 출력 해주세요.
3. 만약 이미지를 식별하기 어려우면 **None**만 출력 해주세요.
4. 만약 한개의 PDF 페이지로 판단되면 필수적인 내역만 추출해서 요약하십시오
    4-1. 일정, 가격, 위치와 같은 세부 내역은 꼭 추출해야 합니다.
    4-2. 필요 없는 내역은 제거해야 합니다.
5. 표로 판단되면 마크다운 문법으로 표현해주세요.
    5-1. 최대한 정확하게 텍스트를 읽어주세요.
중요 **None** or 이미지 요약 문장만 출력 해주세요."""
        
        try:
            summary = self.client.generate_vqa(prompt, image)
            
            # 불필요한 이미지 필터링
            if any(keyword in summary for keyword in ["로고", "아이콘", "버튼", "구분선", "백그라운드", "None"]):
                return None
            return summary.strip()
            
        except Exception as e:
            return None 
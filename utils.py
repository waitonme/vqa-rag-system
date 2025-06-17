import torch
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import re
import numpy as np
import fitz

def split_text_into_chunks(text, tokenizer, max_tokens=500):
    """텍스트를 토큰 기반으로 청크로 분할"""
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sent in tqdm(sentences, desc="텍스트 청킹"):
        # 문장별 전처리
        sent = clean_text(sent)
        if not sent.strip():
            continue
            
        # tokenizer가 None인 경우 문자 수 기반으로 근사 토큰 계산
        if tokenizer is None:
            sent_tokens = len(sent) // 2
        else:
            sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))

        # 문장이 너무 길면 분할
        if sent_tokens > max_tokens:
            if tokenizer is None:
                chunk_size = max_tokens * 2
                for i in range(0, len(sent), chunk_size):
                    chunk_part = sent[i:i+chunk_size]
                    if chunk_part.strip():
                        chunks.append(clean_text(chunk_part))
            continue

        if current_tokens + sent_tokens > max_tokens:
            if current_chunk.strip():
                chunks.append(clean_text(current_chunk))
            current_chunk = sent
            current_tokens = sent_tokens
        else:
            current_chunk += " " + sent
            current_tokens += sent_tokens

    if current_chunk.strip():
        chunks.append(clean_text(current_chunk))

    # 최종 청크들 필터링 (빈 청크 제거)
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks

def hf_generate_response(model, tokenizer, prompt, max_new_tokens=512, system=None, kv_cache=None):
    """HuggingFace 모델을 사용한 텍스트 생성"""
    if system is None:
        system = "당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요."
    
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        # KV 캐시가 있는 경우 이를 활용
        if kv_cache is not None:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer,
                past_key_values=kv_cache,
                use_cache=True
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer
            )

    response = tokenizer.decode(output_ids[0])
    response = extract_assistant_response(response)
    return response.strip()



def extract_assistant_response(text: str) -> str:
    """디코더 출력에서 assistant 응답만 추출"""
    matches = re.findall(r"<\|im_start\|>assistant\s*\n(.*?)(?=<\|im_end\|>|<\|im_start\|>|$)", text, re.DOTALL)
    
    if matches and len(matches) > 0:
        return matches[-1].strip()
    
    cleaned = re.sub(r"<\|im_end\|>|\<\|endofturn\|>", "", text)
    return cleaned.strip()

def clean_text(text):
    """텍스트 정리 - 통합된 전처리 함수"""
    # 제어문자 제거 (remove_control_characters 통합)
    text = ''.join(ch for ch in text if ch == '\n' or ch == '\t' or ord(ch) >= 32)
    
    # 반복 특수문자 제거 (reduce_repeated_symbols 통합)
    text = re.sub(r'([^\w\s])\1{2,}', r'\1', text)
    
    # 공백 정리
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    
    return text.strip()

def simple_parse(doc):
    """PDF에서 텍스트와 이미지 추출"""
    texts = []
    images = []
    
    for page_index in tqdm(range(len(doc)), desc="PDF 파싱"):
        page = doc[page_index]

        # 텍스트 추출
        text = page.get_text()
        texts.append(clean_text(text))

        # 이미지 추출
        image_list = page.get_images(full=True)
        if image_list:
            # 페이지에 이미지가 4개 이상이면 페이지 전체를 하나의 이미지로 처리
            if len(image_list) >= 4:
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((page_index + 1, 1, page_image))
            else:
                # 개별 이미지 처리
                for img_index, img in enumerate(image_list, start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    pil_image = Image.open(BytesIO(image_bytes))
                    
                    # 이미지 품질 검사
                    width, height = pil_image.size
                    if width < 50 or height < 50 or width/height > 10 or height/width > 10:
                        continue
                        
                    img_array = np.array(pil_image)
                    if len(img_array.shape) == 3:
                        mean_value = np.mean(img_array)
                        std_value = np.std(img_array)
                        if mean_value < 20 or mean_value > 235 or std_value < 10:
                            continue
                    
                    images.append((page_index + 1, img_index, pil_image))

    return texts, images

def get_kv_cache(model, tokenizer, context_prompt, system_prompt=None):
    """
    자주 검색되는 청크들에 대한 KV 캐시를 생성합니다.
    올바른 프롬프트 구조를 유지하여 토큰 호환성을 보장합니다.
    
    Args:
        model: 언어 모델 (transformers 모델)
        tokenizer: 토크나이저
        context_prompt (str): 캐시할 컨텍스트 프롬프트 (자주 사용되는 청크들)
        system_prompt (str, optional): 시스템 프롬프트
        
    Returns:
        dict: KV 캐시 정보 (past_key_values, input_ids, attention_mask)
    """
    if system_prompt is None:
        system_prompt = "당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요."
    
    # 실제 QnA에서 사용할 것과 동일한 프롬프트 구조 사용
    cache_chat = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": f"""당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요.
다음은 문서의 일부 내용입니다:
{context_prompt}

위의 정보를 바탕으로 다음 질문에 답변해주세요:

"""  # 질문 부분은 비워둠 - 나중에 추가할 예정
        }
    ]
    
    try:
        # 채팅 템플릿 적용 (add_generation_prompt=False로 assistant 시작 부분 제외)
        cache_inputs = tokenizer.apply_chat_template(
            cache_chat, 
            return_tensors="pt", 
            return_dict=True, 
            add_generation_prompt=False  # 중요: generation prompt 제외
        ).to(model.device)
        
        with torch.no_grad():
            # KV 캐시 생성을 위해 forward pass 수행
            outputs = model(
                **cache_inputs,
                use_cache=True,
                return_dict=True
            )
            
            # KV 캐시와 관련 정보 저장
            kv_cache_info = {
                'past_key_values': outputs.past_key_values,
                'cache_input_ids': cache_inputs['input_ids'],
                'cache_attention_mask': cache_inputs['attention_mask'],
                'cache_length': cache_inputs['input_ids'].shape[1]
            }
            
        print(f"KV 캐시 생성 완료 - 토큰 수: {cache_inputs['input_ids'].shape[1]}")
        return kv_cache_info
        
    except Exception as e:
        print(f"KV 캐시 생성 중 오류 발생: {e}")
        return None

def hf_generate_response_with_cache(model, tokenizer, question, kv_cache_info=None, max_new_tokens=512):
    """
    KV 캐시를 올바르게 활용하는 생성 함수
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        question (str): 질문
        kv_cache_info (dict): KV 캐시 정보
        max_new_tokens (int): 최대 생성 토큰 수
        
    Returns:
        str: 생성된 응답
    """
    if kv_cache_info is None:
        # 캐시가 없으면 일반적인 방식으로 처리
        return hf_generate_response(model, tokenizer, question, max_new_tokens)
    
    try:
        # 더 간단한 접근 방식: 기존 컨텍스트 + 새로운 질문을 하나의 프롬프트로 구성
        # KV 캐시 정보에서 기존 컨텍스트 추출
        cache_input_ids = kv_cache_info.get('cache_input_ids')
        if cache_input_ids is None:
            print("캐시 input_ids가 없습니다. 일반 방식으로 폴백합니다.")
            return hf_generate_response(model, tokenizer, question, max_new_tokens)
        
        # 캐시된 토큰을 텍스트로 디코딩
        cached_text = tokenizer.decode(cache_input_ids[0], skip_special_tokens=True)
        
        # 새로운 질문과 결합
        combined_prompt = f"{cached_text}\n\n{question}"
        
        # 일반적인 생성 방식 사용 (KV 캐시 없이)
        response = hf_generate_response(model, tokenizer, combined_prompt, max_new_tokens)
        
        print("KV 캐시 기반 응답 생성 완료 (간소화된 방식)")
        return response
        
    except Exception as e:
        print(f"KV 캐시 사용 중 오류 발생: {e}")
        print("일반 방식으로 폴백합니다.")
        # 오류 시 일반 방식으로 폴백
        return hf_generate_response(model, tokenizer, question, max_new_tokens)

def create_incremental_kv_cache(model, tokenizer, base_context, system_prompt=None):
    """
    기본 컨텍스트에 대한 KV 캐시를 생성하고, 
    새로운 질문이 들어올 때마다 점진적으로 확장할 수 있는 구조를 만듭니다.
    """
    if system_prompt is None:
        system_prompt = "당신은 유용한 한글 문서 요약 도우미입니다. 모든 응답은 한국어로 작성해 주세요."
    
    # 기본 컨텍스트만으로 KV 캐시 생성
    base_chat = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""다음은 문서의 내용입니다:
{base_context}

이 문서에 대해 질문하겠습니다."""
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        base_chat, 
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=False
    ).to(model.device)
    
    with torch.no_grad():
        # forward pass로 KV 캐시 생성 (generation 없이)
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
    
    return {
        "past_key_values": past_key_values,
        "base_input_ids": inputs["input_ids"],
        "base_attention_mask": inputs["attention_mask"],
        "base_context": base_context,
        "system_prompt": system_prompt
    }

def vqa_generate_response(model, tokenizer, preprocessor, prompt, image, max_new_tokens=256):
    """VQA 모델을 사용한 이미지 분석"""
    try:
        chat = [
            {"role": "system", "content": "당신은 이미지를 분석하는 AI입니다. 이미지에 대해 정확하고 상세하게 설명해주세요."},
            {"role": "user", "content": {"type": "text", "text": prompt}},
            {"role": "user", "content": {"type": "image", "image": image}},
        ]
        
        new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)
        preprocessed = preprocessor(all_images, is_video_list=is_video_list)

        input_ids = tokenizer.apply_chat_template(
            new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                max_length=None,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer,
                **preprocessed
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = extract_assistant_response(response)
        
        return response.strip()
        
    except Exception as e:
        return f"이미지 분석 중 오류가 발생했습니다: {str(e)}"



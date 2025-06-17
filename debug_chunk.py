#!/usr/bin/env python3
"""청크 디버깅 도구"""

from typing import Dict, Any

def print_chunk_debug(file_models: Dict, uploaded_files: Dict) -> str:
    """청크 디버깅 기능 - 전체 청크 내용 출력"""
    if not file_models or not uploaded_files:
        return "업로드된 파일이 없습니다. 먼저 PDF 파일을 업로드해주세요."
    
    debug_output = []
    debug_output.append("**청크 디버깅 정보**")
    debug_output.append("")
    
    for file_id, file_model in file_models.items():
        file_info = uploaded_files[file_id]
        document_name = getattr(file_model, 'document_name', file_info.get('document_name', file_info['filename']))
        
        debug_output.append(f"**문서**: {document_name}")
        debug_output.append(f"- **파일 ID**: `{file_id}`")
        debug_output.append(f"- **총 청크 수**: {len(file_model.chunks) if hasattr(file_model, 'chunks') else 0}개")
        debug_output.append("")
        
        if hasattr(file_model, 'chunks') and file_model.chunks:
            # 텍스트 청크와 이미지 청크 구분
            text_chunks = []
            image_chunks = []
            
            for i, chunk in enumerate(file_model.chunks):
                chunk_str = str(chunk)
                if "이미지" in chunk_str and "문서:" in chunk_str:
                    image_chunks.append((i, chunk_str))
                else:
                    text_chunks.append((i, chunk_str))
            
            # 텍스트 청크 출력
            if text_chunks:
                debug_output.append(f"**텍스트 청크** ({len(text_chunks)}개)")
                debug_output.append("")
                
                for i, (idx, chunk) in enumerate(text_chunks):
                    debug_output.append(f"**청크 {i+1}** (인덱스: {idx})")
                    clean_chunk = chunk.replace("[문서:", "문서:").replace("]", "")
                    debug_output.append(f"```\n{clean_chunk}\n```")
                    debug_output.append(f"**길이**: {len(chunk)} 문자")
                    debug_output.append("---")
                    debug_output.append("")
            
            # 이미지 청크 출력
            if image_chunks:
                debug_output.append(f"**이미지 청크** ({len(image_chunks)}개)")
                debug_output.append("")
                
                for i, (idx, chunk) in enumerate(image_chunks):
                    debug_output.append(f"**이미지 청크 {i+1}** (인덱스: {idx})")
                    clean_chunk = chunk.replace("[이미지", "이미지").replace(" - 문서:", " - 문서:").replace("]", "")
                    debug_output.append(f"```\n{clean_chunk}\n```")
                    debug_output.append(f"**길이**: {len(chunk)} 문자")
                    debug_output.append("---")
                    debug_output.append("")
        else:
            debug_output.append("⚠️ **청크가 없습니다.**")
            debug_output.append("")
        
        # 임베딩 정보
        debug_output.append("**임베딩 정보**")
        if hasattr(file_model, 'chunk_embeddings') and len(file_model.chunk_embeddings) > 0:
            debug_output.append(f"- **임베딩 벡터 수**: {len(file_model.chunk_embeddings)}개")
            if hasattr(file_model.chunk_embeddings, 'shape'):
                debug_output.append(f"- **임베딩 차원**: `{file_model.chunk_embeddings.shape}`")
        else:
            debug_output.append("- **임베딩이 없습니다.**")
        
        debug_output.append("")
        debug_output.append("---")
        debug_output.append("")
    
    debug_output.append("**청크 디버깅 완료**")
    debug_output.append("> **Tip**: 위의 청크 내용을 바탕으로 문서에 대해 질문해보세요!")
    
    return "\n".join(debug_output)

def detect_debug_command(user_message: str) -> bool:
    """디버깅 명령어 감지"""
    debug_keywords = [
        "print chunk", "프린트 청크", "청크 출력", "chunk 출력",
        "print chunks", "show chunks", "청크 보기", "청크 확인"
    ]
    
    return user_message.lower().strip() in debug_keywords 
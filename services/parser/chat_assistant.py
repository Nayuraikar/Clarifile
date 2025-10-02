"""
Chat Assistant Module for Document Generation
Handles detection of document generation requests in chat and processes them.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from services.assistant_generator import generate, export_bytes


def detect_generation_request(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect if a chat query is asking for document generation.
    Returns (document_type, format) or (None, None) if not a generation request.
    """
    if not query:
        return None, None
        
    q_lower = query.lower().strip()
    
    # Define patterns for different document types
    generation_patterns = {
        "flowchart": ["flowchart", "flow chart", "diagram", "mermaid", "process flow", "create flowchart", "make flowchart", "generate flowchart"],
        "short_notes": ["short notes", "bullet points", "summary notes", "brief notes", "quick notes", "create notes", "make notes"],
        "detailed_notes": ["detailed notes", "revision notes", "study notes", "comprehensive notes", "full notes", "create revision", "make revision"],
        "timeline": ["timeline", "chronological", "sequence", "steps", "process timeline", "create timeline", "make timeline"],
        "key_insights": ["key insights", "insights", "takeaways", "main points", "important points", "create insights", "extract insights"],
        "flashcards": ["flashcards", "flash cards", "quiz", "q&a", "questions and answers", "qna", "create flashcards", "make flashcards"]
    }
    
    # Check for downloadable format requests
    format_patterns = {
        "pdf": ["pdf", "downloadable pdf", "download pdf", "as pdf", "in pdf"],
        "docx": ["docx", "word", "doc", "downloadable docx", "download docx", "as docx", "in docx", "as word"],
        "txt": ["txt", "text file", "downloadable txt", "download txt", "downloadable format", "download", "as txt", "in txt"]
    }
    
    detected_kind = None
    detected_format = None
    
    # Detect document type
    for kind, patterns in generation_patterns.items():
        if any(pattern in q_lower for pattern in patterns):
            detected_kind = kind
            break
    
    # Detect format
    for fmt, patterns in format_patterns.items():
        if any(pattern in q_lower for pattern in patterns):
            detected_format = fmt
            break
    
    return detected_kind, detected_format


def process_generation_request(query: str, text: str) -> Dict[str, Any]:
    """
    Process a document generation request and return the appropriate response.
    """
    detected_kind, detected_format = detect_generation_request(query)
    
    if not detected_kind:
        return {"is_generation": False}
    
    try:
        # Generate the requested document
        kind, content = generate(detected_kind, text)
        
        if detected_format:
            # User wants downloadable format
            filename, b64_data = export_bytes(kind, content, detected_format)
            mime_types = {
                'txt': 'text/plain',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'pdf': 'application/pdf'
            }
            mime = mime_types.get(detected_format, 'application/octet-stream')
            
            return {
                "is_generation": True,
                "type": "download",
                "kind": kind,
                "filename": filename,
                "base64": b64_data,
                "mime": mime,
                "content": content,
                "answer": f"I've generated your {kind.replace('_', ' ')} as a downloadable {detected_format.upper()} file. Click the download button to save it."
            }
        else:
            # User wants inline display
            if kind == "flashcards":
                formatted_content = "\n\n".join([f"**Q{i+1}:** {q}\n**A{i+1}:** {a}" for i, (q, a) in enumerate(content)])
            elif kind == "flowchart":
                formatted_content = f"```mermaid\n{content}\n```"
            else:
                formatted_content = str(content)
            
            return {
                "is_generation": True,
                "type": "display",
                "kind": kind,
                "content": content,
                "answer": formatted_content
            }
            
    except Exception as e:
        return {
            "is_generation": True,
            "type": "error",
            "answer": f"I encountered an error generating the {detected_kind.replace('_', ' ')}: {str(e)}"
        }


def enhance_drive_analyze_response(query: str, text: str, original_answer: str, original_score: float) -> Dict[str, Any]:
    """
    Enhance the drive_analyze response with document generation capabilities.
    """
    generation_result = process_generation_request(query, text)
    
    if generation_result["is_generation"]:
        # This is a generation request
        return {
            "answer": generation_result["answer"],
            "score": 1.0,  # High confidence for successful generation
            "assistant": generation_result
        }
    else:
        # Regular Q&A
        return {
            "answer": original_answer,
            "score": original_score,
            "assistant": None
        }

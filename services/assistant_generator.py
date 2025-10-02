from __future__ import annotations

import base64
import io
import re
from typing import Dict, List, Tuple

try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    import requests
    MERMAID_AVAILABLE = True
except Exception:
    MERMAID_AVAILABLE = False

# We will reuse summarization utilities provided in nlp
from services.parser import nlp


def _chunk_if_long(text: str) -> List[str]:
    # Use project smart_chunks for long inputs
    chunks = list(nlp.smart_chunks(text, max_words=600, stride=120)) or []
    return chunks if chunks else [text]


def _normalize_heading(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def clean_text_for_mermaid(text: str) -> str:
    """Clean text to be safe for Mermaid labels"""
    # Remove hidden Unicode characters and normalize
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', '', text)
    # Replace smart quotes and special characters
    text = re.sub(r'[""''`]', '"', text)
    text = re.sub(r'[–—]', '-', text)
    # Remove other problematic characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,;:()\[\]{}!?&]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_flowchart_mermaid(text: str) -> str:
    """Generate a proper Mermaid flowchart from text content"""
    # Clean the input text first
    text = clean_text_for_mermaid(text)
    
    # Try to extract meaningful structure from the content
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    nodes: List[str] = []
    
    # Look for structured content (headings, bullet points, etc.)
    for line in lines[:10]:  # Limit to first 10 lines
        # Skip very short lines
        if len(line.split()) < 2:
            continue
        
        # Clean and truncate the line for use as a node
        clean_line = clean_text_for_mermaid(line)
        
        # Remove bullet points and numbering
        clean_line = re.sub(r'^[\-\*\•\d\.\)]+\s*', '', clean_line)
        
        # Ensure reasonable length (20-40 chars is good for Mermaid)
        if len(clean_line) > 40:
            # Try to find a good breaking point
            words = clean_line.split()
            if len(words) > 4:
                clean_line = ' '.join(words[:4]) + '...'
            else:
                clean_line = clean_line[:37] + '...'
        
        if len(clean_line) >= 3:  # Minimum meaningful length
            nodes.append(clean_line)
        
        if len(nodes) >= 6:  # Limit number of nodes
            break
    
    # If no good structure found, create a simple process flow
    if len(nodes) < 2:
        # Try to extract key concepts
        words = text.split()
        if 'trip' in text.lower() or 'travel' in text.lower():
            nodes = ["Plan Trip", "Visit Destinations", "Complete Activities"]
        elif 'notes' in text.lower() or 'todo' in text.lower():
            nodes = ["Review Notes", "Complete Tasks", "Finish Goals"]
        else:
            nodes = ["Start", "Process Content", "Complete"]
    
    # Generate clean Mermaid syntax
    mermaid = ["flowchart TD"]
    
    for i, node in enumerate(nodes):
        # Final cleaning for Mermaid
        label = clean_text_for_mermaid(node)
        
        # Ensure label is not empty
        if not label:
            label = f"Step {i+1}"
        
        # Escape quotes in labels
        label = label.replace('"', "'")
        
        # Add the node
        mermaid.append(f'  A{i}["{label}"]')
    
    # Add connections
    for i in range(len(nodes) - 1):
        mermaid.append(f"  A{i} --> A{i+1}")
    
    return "\n".join(mermaid)


def generate_short_notes(text: str) -> str:
    # ~1-page bullets
    chunks = _chunk_if_long(text)
    notes: List[str] = []
    for ch in chunks[:4]:
        s = nlp.summarize_with_gemini(ch, max_tokens=220) or ch[:1000]
        # Clean any emojis
        s = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', s)
        notes.append(s)
        if len("\n\n".join(notes)) > 1200:
            break
    return "\n\n".join(f"- {p.strip()}" for p in notes if p.strip())


def generate_detailed_notes(text: str) -> str:
    # 2–3 pages structured
    chunks = _chunk_if_long(text)
    out: List[str] = ["## Overview"]
    overview = nlp.summarize_with_gemini(" ".join(chunks[:2]), max_tokens=380) or text[:1800]
    overview = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', overview)
    out.append(overview)
    out.append("\n## Key Sections")
    for idx, ch in enumerate(chunks[:6], 1):
        head = _normalize_heading(f"Section {idx}")
        sec = nlp.summarize_with_gemini(ch, max_tokens=320) or ch[:1200]
        sec = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', sec)
        out.append(f"\n### {head}\n{sec}")
    out.append("\n## Takeaways")
    take = nlp.summarize_with_gemini(text, max_tokens=180) or ""
    take = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', take)
    out.append("- " + take.replace("\n", " ").strip())
    return "\n".join(out)


def generate_timeline(text: str) -> str:
    # Simplified: extract likely chronological bullets
    base = nlp.summarize_with_gemini(text, max_tokens=300) or text[:1500]
    base = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', base)
    lines = [l.strip(" -\t") for l in base.split("\n") if len(l.split()) >= 3]
    lines = lines[:12]
    # Represent as markdown timeline
    return "\n".join(f"- [Step {i+1}] {l}" for i, l in enumerate(lines))


def generate_key_insights(text: str) -> str:
    s = nlp.summarize_with_gemini(text, max_tokens=260) or text[:1200]
    s = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', s)
    bullets = [b.strip() for b in re.split(r"[\n\-]+", s) if len(b.strip().split()) >= 3]
    bullets = bullets[:10]
    return "\n".join(f"- {b}" for b in bullets)


def generate_flashcards(text: str) -> List[Tuple[str, str]]:
    # Generate exactly 3 meaningful Q&A pairs
    chunks = _chunk_if_long(text)
    cards: List[Tuple[str, str]] = []
    
    # Try to generate specific questions from content
    for i, chunk in enumerate(chunks[:3]):
        summary = nlp.summarize_with_gemini(chunk, max_tokens=200) or chunk[:800]
        summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', summary)
        
        # Generate different types of questions
        if i == 0:
            question = "What is the main topic or purpose of this content?"
            answer = summary.split('.')[0] + '.' if '.' in summary else summary[:100]
        elif i == 1:
            question = "What are the key points or methods discussed?"
            # Extract key points
            sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 10]
            answer = sentences[0] + '.' if sentences else summary[:100]
        else:
            question = "What conclusions or insights can be drawn?"
            # Get last meaningful sentence
            sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 10]
            answer = sentences[-1] + '.' if sentences else summary[:100]
        
        if len(answer.split()) >= 3:
            cards.append((question, answer))
    
    # Ensure we have exactly 3 cards
    while len(cards) < 3:
        if len(cards) == 0:
            cards.append(("What is this content about?", text[:200] + "..."))
        elif len(cards) == 1:
            cards.append(("What are the important details?", text[200:400] + "..."))
        else:
            cards.append(("What should you remember?", "Key takeaways from the content."))
    
    return cards[:3]  # Ensure exactly 3 cards


def render_markdown(kind: str, content) -> str:
    if kind == "flashcards":
        lines = ["## Q&A Flashcards"]
        for i, (q, a) in enumerate(content, 1):
            lines.append(f"\n### Q{i}: {q}\nA{i}: {a}")
        return "\n".join(lines)
    if kind == "flowchart":
        return f"## Flowchart (Mermaid)\n\n```mermaid\n{content}\n```"
    return str(content)


def create_simple_flowchart() -> str:
    """Create a simple, guaranteed valid Mermaid flowchart"""
    return """flowchart TD
  A0[Start]
  A1[Process]
  A2[End]
  A0 --> A1
  A1 --> A2"""

def mermaid_to_image(mermaid_code: str) -> bytes:
    """Convert Mermaid code to PNG image using Mermaid.ink API"""
    try:
        print(f"DEBUG: Converting Mermaid to image:")
        print(f"DEBUG: Mermaid code: {mermaid_code}")
        
        # Validate and clean the Mermaid code first
        lines = [line.strip() for line in mermaid_code.strip().split('\n') if line.strip()]
        
        # Ensure it starts with flowchart
        if not lines[0].startswith('flowchart'):
            print("DEBUG: Invalid Mermaid syntax - missing flowchart declaration")
            mermaid_code = create_simple_flowchart()
        
        # Check for basic syntax issues
        if len(lines) < 3:  # Need at least flowchart declaration + 1 node + 1 connection
            print("DEBUG: Mermaid code too short, using simple fallback")
            mermaid_code = create_simple_flowchart()
        
        # First try with the provided code
        import urllib.parse
        encoded = urllib.parse.quote(mermaid_code.encode('utf-8'))
        
        # Use Mermaid.ink API to generate image
        url = f"https://mermaid.ink/img/{encoded}"
        print(f"DEBUG: Mermaid.ink URL: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"DEBUG: Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("DEBUG: Successfully generated image from Mermaid.ink")
            return response.content
        else:
            print(f"DEBUG: Mermaid.ink failed with original code, trying simple fallback")
            # Try with simple flowchart
            simple_code = create_simple_flowchart()
            encoded_simple = urllib.parse.quote(simple_code.encode('utf-8'))
            url_simple = f"https://mermaid.ink/img/{encoded_simple}"
            
            response_simple = requests.get(url_simple, timeout=10)
            if response_simple.status_code == 200:
                print("DEBUG: Successfully generated simple flowchart image")
                return response_simple.content
            else:
                print(f"DEBUG: Even simple flowchart failed, using text image")
                return create_text_image(mermaid_code)
    except Exception as e:
        print(f"DEBUG: Exception in mermaid_to_image: {e}")
        return create_text_image(mermaid_code)

def create_text_image(text: str) -> bytes:
    """Create a simple text-based image as fallback"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text
        lines = text.split('\n')[:20]  # Limit lines
        y = 20
        for line in lines:
            draw.text((20, y), line[:80], fill='black', font=font)
            y += 25
        
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        print(f"DEBUG: PIL not available or failed: {e}")
        # Create a minimal valid PNG (1x1 white pixel)
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

def export_bytes(kind: str, content, fmt: str) -> Tuple[str, str]:
    """Return (filename, base64_data)."""
    fmt = (fmt or "txt").lower()
    base_name = f"assistant_{kind}"

    # Special handling for flowcharts - convert to image
    if kind == "flowchart" and fmt in {"png", "image", "img"}:
        image_data = mermaid_to_image(content)
        return f"{base_name}.png", base64.b64encode(image_data).decode("ascii")
    
    # For flowcharts with other formats, also try to include image
    if kind == "flowchart" and fmt in {"pdf", "docx"}:
        # For now, fall back to text representation
        pass

    if fmt in {"txt"}:
        text = render_markdown(kind, content)
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        data = text.encode("utf-8")
        return f"{base_name}.txt", base64.b64encode(data).decode("ascii")

    if fmt in {"docx"} and DOCX_AVAILABLE:
        doc = Document()
        text = render_markdown(kind, content)
        for para in text.split("\n\n"):
            doc.add_paragraph(para)
        buf = io.BytesIO()
        doc.save(buf)
        return f"{base_name}.docx", base64.b64encode(buf.getvalue()).decode("ascii")

    if fmt in {"pdf"} and PDF_AVAILABLE:
        text = render_markdown(kind, content)
        buf = io.BytesIO()
        can = canvas.Canvas(buf, pagesize=LETTER)
        width, height = LETTER
        y = height - 72
        for line in text.split("\n"):
            if y < 72:
                can.showPage()
                y = height - 72
            can.drawString(72, y, line[:1000])
            y -= 14
        can.save()
        return f"{base_name}.pdf", base64.b64encode(buf.getvalue()).decode("ascii")

    # Fallback to txt if format unsupported or libs missing
    text = render_markdown(kind, content)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    data = text.encode("utf-8")
    return f"{base_name}.txt", base64.b64encode(data).decode("ascii")


def generate(kind: str, text: str):
    kind = (kind or "").lower()
    if kind in {"flowchart", "diagram", "mermaid"}:
        return "flowchart", generate_flowchart_mermaid(text)
    if kind in {"short_notes", "short", "bullets"}:
        return "short_notes", generate_short_notes(text)
    if kind in {"detailed_notes", "detailed", "revision"}:
        return "detailed_notes", generate_detailed_notes(text)
    if kind in {"timeline", "timelines"}:
        return "timeline", generate_timeline(text)
    if kind in {"key_insights", "insights", "takeaways"}:
        return "key_insights", generate_key_insights(text)
    if kind in {"flashcards", "qa", "qna"}:
        return "flashcards", generate_flashcards(text)
    # Default to short notes
    return "short_notes", generate_short_notes(text)



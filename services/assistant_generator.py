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

def extract_key_concepts(text: str, max_concepts: int = 5) -> list:
    """Extract key concepts from text for use in flowcharts"""
    from collections import Counter
    import re
    
    # Remove code blocks and special characters
    clean_text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
    clean_text = re.sub(r'[^\w\s-]', ' ', clean_text)  # Keep only words, spaces, and hyphens
    
    # Tokenize and count words (excluding stop words)
    words = re.findall(r'\b\w{3,}\b', clean_text.lower())
    stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'which', 'their', 'they', 'them', 'then', 'there', 'were', 'when', 'what', 'where'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Get most common words
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(max_concepts)]

def generate_flowchart_mermaid(text: str) -> str:
    """Generate a Mermaid flowchart based on the actual content of the analyzed file"""
    # Clean the input text first
    text = clean_text_for_mermaid(text)
    if not text.strip():
        return 'flowchart TD\n    A[No content to generate flowchart]'
    
    # Extract key concepts from the text
    concepts = extract_key_concepts(text, max_concepts=5)
    
    # Start building the flowchart
    mermaid = ["flowchart TD"]
    
    # Add start node
    mermaid.append('    A[Start]')
    
    # Add concept nodes
    prev_node = 'A'
    for i, concept in enumerate(concepts, 1):
        node_id = chr(65 + i)  # B, C, D, etc.
        mermaid.append(f'    {prev_node} --> {node_id}[{concept.capitalize()}]')
        prev_node = node_id
    
    # Add end node
    mermaid.append(f'    {prev_node} --> Z[End]')
    
    # Add some cross-connections for better visualization
    if len(concepts) >= 3:
        mermaid.append(f'    {chr(65 + 1)} --> {chr(65 + 3)}')  # B -> D
    if len(concepts) >= 5:
        mermaid.append(f'    {chr(65 + 2)} --> {chr(65 + 4)}')  # C -> E
    
    return '\n'.join(mermaid)


def generate_short_notes(text: str) -> str:
    """Generate revision notes following the point-wise structure with sections"""
    chunks = _chunk_if_long(text)
    
    # Extract key concepts and organize into sections
    all_content = " ".join(chunks[:3])  # Use first 3 chunks for analysis
    
    # Generate structured notes
    notes = ["# Revision Notes"]
    
    # Key Concepts section
    notes.append("\n## Key Concepts")
    key_concepts = nlp.summarize_with_gemini(all_content, max_tokens=200) or all_content[:800]
    key_concepts = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', key_concepts)
    
    # Convert to bullet points (max 10-15 points)
    concept_points = []
    sentences = [s.strip() for s in key_concepts.split('.') if len(s.strip()) > 10]
    for sentence in sentences[:8]:  # Max 8 key concepts
        if sentence:
            concept_points.append(f"* {sentence.strip()}.")
    
    if not concept_points:
        concept_points = [f"* {all_content[:100]}..."]
    
    notes.extend(concept_points)
    
    # Definitions section (if applicable)
    if any(word in text.lower() for word in ['definition', 'define', 'means', 'is a', 'refers to']):
        notes.append("\n## Definitions")
        # Extract definition-like content
        definition_content = nlp.summarize_with_gemini(f"Extract definitions from: {all_content}", max_tokens=150) or ""
        definition_content = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', definition_content)
        
        def_sentences = [s.strip() for s in definition_content.split('.') if len(s.strip()) > 10]
        for sentence in def_sentences[:5]:  # Max 5 definitions
            if sentence:
                notes.append(f"* {sentence.strip()}.")
    
    # Examples section (if applicable)
    if any(word in text.lower() for word in ['example', 'instance', 'case', 'such as']):
        notes.append("\n## Examples")
        example_content = nlp.summarize_with_gemini(f"Extract examples from: {all_content}", max_tokens=120) or ""
        example_content = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', example_content)
        
        example_sentences = [s.strip() for s in example_content.split('.') if len(s.strip()) > 10]
        for sentence in example_sentences[:4]:  # Max 4 examples
            if sentence:
                notes.append(f"* {sentence.strip()}.")
    
    # Important Points section
    notes.append("\n## Important Points")
    important_content = nlp.summarize_with_gemini(f"Extract most important points from: {all_content}", max_tokens=180) or all_content[:600]
    important_content = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', important_content)
    
    important_sentences = [s.strip() for s in important_content.split('.') if len(s.strip()) > 10]
    for sentence in important_sentences[:6]:  # Max 6 important points
        if sentence:
            notes.append(f"* {sentence.strip()}.")
    
    return "\n".join(notes)


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
    """Generate a timeline with 5-7 chronological milestones or logical steps"""
    chunks = _chunk_if_long(text)
    all_content = " ".join(chunks[:3])
    
    # Generate timeline-focused summary
    timeline_content = nlp.summarize_with_gemini(f"Extract chronological steps or progression from: {all_content}", max_tokens=250) or all_content[:1000]
    timeline_content = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', timeline_content)
    
    # Create timeline structure
    timeline = ["# Timeline"]
    
    # Detect content type and create appropriate timeline
    content_lower = all_content.lower()
    
    if any(word in content_lower for word in ['history', 'evolution', 'development', 'progress']):
        # Historical/Evolution timeline
        timeline.append("\n## Historical Development")
        
        # Extract key periods or developments
        sentences = [s.strip() for s in timeline_content.split('.') if len(s.strip()) > 15]
        milestones = []
        
        for i, sentence in enumerate(sentences[:7]):
            if sentence:
                # Try to extract year/date if present
                year_match = re.search(r'\b(19|20)\d{2}\b', sentence)
                if year_match:
                    year = year_match.group()
                    milestone_text = sentence.replace(year, '').strip()
                    milestones.append(f"**{year}** – {milestone_text}")
                else:
                    milestones.append(f"**Stage {i+1}** – {sentence}")
        
        if not milestones:
            milestones = [
                "**Early Stage** – Initial concepts and foundations",
                "**Development** – Key innovations and improvements", 
                "**Maturation** – Widespread adoption and refinement",
                "**Modern Era** – Current applications and future directions"
            ]
    
    elif any(word in content_lower for word in ['algorithm', 'process', 'method', 'procedure']):
        # Process/Algorithm timeline
        timeline.append("\n## Process Steps")
        
        # Extract process steps
        process_summary = nlp.summarize_with_gemini(f"List the main steps in order from: {all_content}", max_tokens=200) or ""
        process_summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', process_summary)
        
        steps = [s.strip() for s in process_summary.split('.') if len(s.strip()) > 10]
        milestones = []
        
        for i, step in enumerate(steps[:6]):
            if step:
                milestones.append(f"**Step {i+1}** – {step}")
        
        if not milestones:
            milestones = [
                "**Step 1 – Initialization** – Set up initial conditions and variables",
                "**Step 2 – Input Processing** – Read and validate input data",
                "**Step 3 – Main Algorithm** – Execute core processing logic",
                "**Step 4 – Result Generation** – Compute and format results",
                "**Step 5 – Output** – Return or display final results"
            ]
    
    elif any(word in content_lower for word in ['learn', 'study', 'education', 'course']):
        # Learning/Study timeline
        timeline.append("\n## Learning Progression")
        
        learning_steps = [
            "**Foundation** – Understanding basic concepts and terminology",
            "**Core Learning** – Mastering fundamental principles and methods", 
            "**Application** – Practicing with examples and exercises",
            "**Integration** – Connecting concepts with broader knowledge",
            "**Mastery** – Achieving proficiency and problem-solving ability",
            "**Advanced Topics** – Exploring complex applications and extensions"
        ]
        milestones = learning_steps[:6]
    
    else:
        # Generic timeline based on content structure
        timeline.append("\n## Key Milestones")
        
        # Extract key points and organize chronologically
        key_points = [s.strip() for s in timeline_content.split('.') if len(s.strip()) > 15]
        milestones = []
        
        for i, point in enumerate(key_points[:7]):
            if point:
                milestones.append(f"**Milestone {i+1}** – {point}")
        
        # Ensure we have at least 5 milestones
        while len(milestones) < 5:
            milestone_num = len(milestones) + 1
            if milestone_num == 1:
                milestones.append("**Beginning** – Initial setup and preparation")
            elif milestone_num == 2:
                milestones.append("**Development** – Core implementation and progress")
            elif milestone_num == 3:
                milestones.append("**Refinement** – Improvements and optimization")
            elif milestone_num == 4:
                milestones.append("**Integration** – Combining components and testing")
            else:
                milestones.append("**Completion** – Final results and conclusions")
    
    # Add milestones to timeline (ensure 5-7 items)
    timeline.extend(milestones[:7])
    
    return "\n".join(timeline)


def generate_key_insights(text: str) -> str:
    s = nlp.summarize_with_gemini(text, max_tokens=260) or text[:1200]
    s = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', s)
    bullets = [b.strip() for b in re.split(r"[\n\-]+", s) if len(b.strip().split()) >= 3]
    bullets = bullets[:10]
    return "\n".join(f"- {b}" for b in bullets)


def generate_flashcards(text: str) -> List[Tuple[str, str]]:
    """Generate 3-5 flashcards with mixed question types (definition, concept, application)"""
    chunks = _chunk_if_long(text)
    cards: List[Tuple[str, str]] = []
    all_content = " ".join(chunks[:3])
    
    # Generate content-aware summary for better Q&A
    content_summary = nlp.summarize_with_gemini(all_content, max_tokens=300) or all_content[:1000]
    content_summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', content_summary)
    
    # Card 1: Definition type question
    if any(word in text.lower() for word in ['data structure', 'algorithm', 'definition', 'what is', 'means']):
        if 'data structure' in text.lower():
            cards.append((
                "What is a Data Structure?",
                "A way of organizing and storing data for efficient access and modification."
            ))
        elif 'algorithm' in text.lower():
            cards.append((
                "What is an Algorithm?", 
                "A step-by-step procedure for solving problems and performing operations efficiently."
            ))
        else:
            # Extract first definition-like sentence
            sentences = [s.strip() for s in content_summary.split('.') if len(s.strip()) > 15]
            if sentences:
                cards.append((
                    "What is the main concept discussed?",
                    sentences[0] + "."
                ))
    
    # Card 2: Concept/Key points question
    concept_summary = nlp.summarize_with_gemini(f"What are the key concepts in: {all_content}", max_tokens=150) or ""
    concept_summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', concept_summary)
    
    if concept_summary:
        key_points = [s.strip() for s in concept_summary.split('.') if len(s.strip()) > 10]
        if key_points:
            cards.append((
                "What are the key points or main ideas?",
                key_points[0] + "." if key_points[0] else "Key concepts and important principles."
            ))
    
    # Card 3: Application question
    if any(word in text.lower() for word in ['use', 'application', 'example', 'how', 'why']):
        app_summary = nlp.summarize_with_gemini(f"How is this used or applied: {all_content}", max_tokens=120) or ""
        app_summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', app_summary)
        
        if app_summary:
            app_sentences = [s.strip() for s in app_summary.split('.') if len(s.strip()) > 10]
            if app_sentences:
                cards.append((
                    "How is this concept used or applied?",
                    app_sentences[0] + "."
                ))
    
    # Card 4: Importance/Why question
    if len(cards) < 4:
        importance_summary = nlp.summarize_with_gemini(f"Why is this important: {all_content}", max_tokens=100) or ""
        importance_summary = re.sub(r'[^\w\s\-.,;:()\[\]{}!?\'"\n]', '', importance_summary)
        
        if importance_summary:
            imp_sentences = [s.strip() for s in importance_summary.split('.') if len(s.strip()) > 10]
            if imp_sentences:
                cards.append((
                    "Why is this concept important?",
                    imp_sentences[0] + "."
                ))
    
    # Card 5: Context/Background question (if we have enough content)
    if len(cards) < 5 and len(chunks) > 2:
        context_sentences = [s.strip() for s in content_summary.split('.') if len(s.strip()) > 15]
        if len(context_sentences) > 2:
            cards.append((
                "What background or context is important to understand?",
                context_sentences[-1] + "."
            ))
    
    # Ensure we have at least 3 cards with fallbacks
    while len(cards) < 3:
        if len(cards) == 0:
            cards.append((
                "What is the main topic of this content?",
                content_summary[:150] + "..." if len(content_summary) > 150 else content_summary
            ))
        elif len(cards) == 1:
            cards.append((
                "What are the important details mentioned?",
                all_content[200:350] + "..." if len(all_content) > 350 else all_content[200:]
            ))
        else:
            cards.append((
                "What should you remember from this content?",
                "Key takeaways and important concepts for understanding the topic."
            ))
    
    # Return 3-5 cards (prefer 4-5 if we have good content)
    return cards[:5]


def render_markdown(kind: str, content) -> str:
    if kind == "flashcards":
        lines = ["# Flashcards"]
        for i, (q, a) in enumerate(content, 1):
            lines.append(f"\n**Flashcard {i}**")
            lines.append(f"* Q: {q}")
            lines.append(f"* A: {a}")
        return "\n".join(lines)
    if kind == "flowchart":
        return f"# Flowchart\n\n```mermaid\n{content}\n```"
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
        
        # Ensure it starts with flowchart and has content
        if not lines or not lines[0].startswith('flowchart'):
            print("DEBUG: Invalid Mermaid syntax - missing flowchart declaration")
            mermaid_code = create_simple_flowchart()
            lines = [line.strip() for line in mermaid_code.strip().split('\n') if line.strip()]
        
        # Check for basic syntax issues
        if len(lines) < 3:  # Need at least flowchart declaration + 1 node + 1 connection
            print("DEBUG: Mermaid code too short, using simple fallback")
            mermaid_code = create_simple_flowchart()
        
        # Clean up the mermaid code - remove any extra backticks or formatting
        mermaid_code = mermaid_code.strip()
        if mermaid_code.startswith('```mermaid'):
            mermaid_code = mermaid_code.replace('```mermaid', '').replace('```', '').strip()
        
        print(f"DEBUG: Cleaned Mermaid code: {mermaid_code}")
        
        # First try with the provided code
        import base64
        
        # FIX: Mermaid.ink expects base64-url encoded string, not quote()
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        
        # Use Mermaid.ink API to generate image
        url = f"https://mermaid.ink/img/{encoded}"
        print(f"DEBUG: Final Mermaid.ink URL: {url}")
        
        response = requests.get(url, timeout=15)
        print(f"DEBUG: Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("DEBUG: Successfully generated image from Mermaid.ink")
            return response.content
        else:
            print(f"DEBUG: Mermaid.ink failed with original code, trying simple fallback")
            # Try with simple flowchart
            simple_code = create_simple_flowchart()
            encoded_simple = base64.urlsafe_b64encode(simple_code.encode("utf-8")).decode("utf-8")
            url_simple = f"https://mermaid.ink/img/{encoded_simple}"
            
            response_simple = requests.get(url_simple, timeout=15)
            if response_simple.status_code == 200:
                print("DEBUG: Successfully generated simple flowchart image")
                return response_simple.content
            else:
                print(f"DEBUG: Even simple flowchart failed, creating text fallback")
                return create_text_image("Flowchart generation failed")
    except Exception as e:
        print(f"DEBUG: Exception in mermaid_to_image: {e}")
        return create_text_image(f"Error: {str(e)}")

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



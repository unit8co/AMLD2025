import re
import uuid
from typing import List

def slugify(text: str, max_len=32) -> str:
    """Naive slugify for node IDs."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len] or str(uuid.uuid4())

def read_markdown(path: str) -> str:
    """Reads the entire contents of a Markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with robust abbreviation handling"""
    abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Jr.', 'Sr.', 'No.', 'St.', 'Ave.', 'Rd.'}
    
    # Simplified regex pattern without lookbehind
    pattern = r'(?<=[.!?])\s+(?=[A-Z"“])'
    
    # Initial split
    sentences = re.split(pattern, text)
    
    # Merge sentences that end with known abbreviations
    merged = []
    i = 0
    while i < len(sentences):
        if i < len(sentences)-1 and any(sentences[i].endswith(abbr) for abbr in abbreviations):
            merged.append(sentences[i] + " " + sentences[i+1].lstrip())
            i += 2
        else:
            merged.append(sentences[i])
            i += 1
    
    return [s.strip() for s in merged if s.strip()]

def chunk_text(text: str, max_tokens: int = 1500, overlap_tokens: int = 500) -> List[str]:
    """
    Chunks text into overlapping segments, respecting sentence and paragraph boundaries.
    
    Args:
        text: The text to chunk
        max_tokens: Target tokens per chunk (approximate, default increased to 1500 for better context)
        overlap_tokens: Target tokens to overlap between chunks (approximate, increased to 500 for better continuity)
    
    Returns:
        List of text chunks with overlap
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for paragraph in paragraphs:
        # Rough token estimation
        para_tokens = len(paragraph) // 4
        
        if current_tokens + para_tokens <= max_tokens * 1.2:  # Allow 20% overflow
            # Paragraph fits in current chunk
            current_chunk.append(paragraph)
            current_tokens += para_tokens
        else:
            # If paragraph is very large, split it into sentences
            if para_tokens > max_tokens:
                sentences = split_into_sentences(paragraph)
                sentence_chunk = []
                sentence_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens_estimate = len(sentence) // 4
                    
                    if sentence_tokens + sentence_tokens_estimate <= max_tokens * 1.2:
                        sentence_chunk.append(sentence)
                        sentence_tokens += sentence_tokens_estimate
                    else:
                        if sentence_chunk:
                            chunks.append(" ".join(sentence_chunk))
                        sentence_chunk = [sentence]
                        sentence_tokens = sentence_tokens_estimate
                
                if sentence_chunk:
                    chunks.append(" ".join(sentence_chunk))
            else:
                # Flush current chunk if it exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                # Start new chunk with this paragraph
                current_chunk = [paragraph]
                current_tokens = para_tokens
    
    # Add the final chunk if it exists
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    # Add overlap by including the end of previous chunk at the start of next chunk
    chunks_with_overlap = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            chunks_with_overlap.append(chunk)
            continue
            
        # Get overlap from end of previous chunk
        prev_chunk = chunks[i-1]
        prev_sentences = split_into_sentences(prev_chunk)
        overlap_text = ""
        overlap_tokens_count = 0
        
        for sentence in reversed(prev_sentences):
            sentence_tokens = len(sentence) // 4
            if overlap_tokens_count + sentence_tokens > overlap_tokens:
                break
            overlap_text = sentence + " " + overlap_text
            overlap_tokens_count += sentence_tokens
            
        if overlap_text:
            chunk = overlap_text.strip() + "\n\n" + chunk
            
        chunks_with_overlap.append(chunk)
    
    return chunks_with_overlap 

def unique_everseen(iterable):
    """List unique elements in order they appear"""
    seen = set()
    return [x for x in iterable if not (x in seen or seen.add(x))] 
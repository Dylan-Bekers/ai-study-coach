from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    import re

    # Split on sentence boundaries (period, exclamation, question mark followed by whitespace)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds chunk_size, split it hard
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    chunks.append(sentence[i:i + chunk_size])
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def load_all_pdfs_from_folder(folder_path):
    folder = Path(folder_path)
    documents = []

    for pdf_file in folder.glob("*.pdf"):
        text = extract_text_from_pdf(pdf_file)
        documents.append({
            "filename": pdf_file.name,
            "text": text
        })

    return documents
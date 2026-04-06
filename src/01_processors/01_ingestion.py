import os
import re
import logging
import pdfplumber
import docx
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.Ingestion")

class ResumeIngestor:
    def __init__(self):
        self.major_anchors = {
            "SUMMARY": ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE", "ABOUT ME"],
            "SKILLS": ["SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES"],
            "EXPERIENCE": ["EXPERIENCE", "WORK HISTORY", "PROFESSIONAL EXPERIENCE"],
            "PROJECTS": ["PROJECTS", "TECHNICAL PROJECTS"],
            "EDUCATION": ["EDUCATION", "ACADEMIC BACKGROUND"],
            "CERTIFICATIONS": ["CERTIFICATIONS", "AWARDS"]
        }

        self.spatial_threshold = 14 
        self.date_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|20\d{2})\b'

    def _get_anchor_label(self, text: str, is_bold: bool) -> Optional[str]:
        """ Identifies major sections """
        clean = text.strip().upper().rstrip(':')
        words = clean.split()
        if is_bold and 0 < len(words) < 5:
            for section, keywords in self.major_anchors.items():
                if any(k == clean or k in clean for k in keywords):
                    return section
        return None

    def _is_sub_header(self, text: str, is_bold: bool, section: str, gap: float) -> bool:
        """Heuristic to detect a Job or Project header vs a regular bold keyword."""
        if not is_bold or text.startswith(('●', '•', '-', '*')):
            return False
        
        has_divider = '|' in text
        has_date = bool(re.search(self.date_pattern, text))
        
        if section == "EXPERIENCE":
            return has_date or has_divider
        
        if section == "PROJECTS":
            return has_divider or (gap > self.spatial_threshold)
            
        return False

    def _extract_pdf(self, path: Path) -> List[Dict]:
        chunks = []
        current_section = "CONTACT_HEADER"
        current_content = []
        last_y_bottom = 0
        anchors_started = False

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                lines = page.extract_text_lines(layout=True)
                for line in lines:
                    text = line['text'].strip()
                    if not text: continue

                    char_meta = line['chars'][0]
                    font_name = char_meta.get('fontname', '').lower()
                    is_bold = any(x in font_name for x in ['bold', 'black', 'heavy'])
                    
                    anchor_label = self._get_anchor_label(text, is_bold)
                    gap = line['top'] - last_y_bottom

                    if anchor_label:
                        if current_content:
                            chunks.append({"section": current_section, "content": "\n".join(current_content)})
                        current_section = anchor_label
                        current_content = []
                        anchors_started = True
                    else:
                        if anchors_started and self._is_sub_header(text, is_bold, current_section, gap):
                            current_content.append("---ITEM_SPLIT---")
                        
                        current_content.append(text)
                    last_y_bottom = line['bottom']

        chunks.append({"section": current_section, "content": "\n".join(current_content)})
        return chunks

    def _extract_docx(self, path: Path) -> List[Dict]:
        doc = docx.Document(path)
        chunks = []
        current_section = "CONTACT_HEADER"
        current_content = []
        anchors_started = False

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text: continue
            
            is_bold = any(run.bold for run in para.runs)
            anchor_label = self._get_anchor_label(text, is_bold)

            if anchor_label:
                if current_content:
                    chunks.append({"section": current_section, "content": "\n".join(current_content)})
                current_section = anchor_label
                current_content = []
                anchors_started = True
            else:
                if anchors_started and self._is_sub_header(text, is_bold, current_section, 20): 
                    current_content.append("---ITEM_SPLIT---")
                current_content.append(text)

        chunks.append({"section": current_section, "content": "\n".join(current_content)})
        return chunks

    def process(self, file_path: str) -> List[Dict]:
        path = Path(file_path)
        if not path.exists(): return []

        if path.suffix.lower() == '.pdf':
            raw_chunks = self._extract_pdf(path)
        elif path.suffix.lower() == '.docx':
            raw_chunks = self._extract_docx(path)
        else:
            return []

        final_assets = []
        for chunk in raw_chunks:
            content = chunk['content']
            if "---ITEM_SPLIT---" in content:
                parts = content.split("---ITEM_SPLIT---")
                for p in parts:
                    clean_p = p.strip()
                    if len(clean_p) > 20:
                        first_line = clean_p.split('\n')[0]
                        sub_label = first_line.split('|')[0].strip() if '|' in first_line else first_line[:40]
                        final_assets.append({
                            "section": f"{chunk['section']}: {sub_label}",
                            "content": clean_p,
                            "metadata": {"source": path.name}
                        })
            else:
                if len(content.strip()) > 5:
                    final_assets.append({
                        "section": chunk['section'],
                        "content": content.strip(),
                        "metadata": {"source": path.name}
                    })
        return final_assets

if __name__ == "__main__":
    ingestor = ResumeIngestor()
    path = "/Users/kvenkateshrao/Library/Mobile Documents/com~apple~CloudDocs/Venkatesh_Kastala_Software_Engineer.pdf"
    results = ingestor.process(path)
    for i, c in enumerate(results):
        print(f"[{i+1}] {c['section']}\n{c['content'][:200]}...\n")
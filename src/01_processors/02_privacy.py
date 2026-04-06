import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.Privacy")

class PrivacyScrubber:
    def __init__(self):

        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b',
            "LINKEDIN": r'https?://(?:www\.)?linkedin\.com/in/[\w\-\!@#$%^&*()]+/?',
        }

    def _extract_pii_seeds(self, contact_chunk: str) -> Dict[str, str]:
        seeds = {}
        lines = contact_chunk.strip().split('\n')
        if lines:
            seeds["NAME"] = lines[0].strip()
        
        email_match = re.search(self.patterns["EMAIL"], contact_chunk)
        if email_match:
            seeds["EMAIL_ADDR"] = email_match.group()
            
        return seeds

    def scrub(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        contact_content = chunks[0]['content'] if chunks[0]['section'] == "CONTACT_HEADER" else ""
        seeds = self._extract_pii_seeds(contact_content)
        
        scrubbed_data = []
        for chunk in chunks:
            text = chunk['content']
            
            if "NAME" in seeds:
                text = re.sub(re.escape(seeds["NAME"]), "{{USER_NAME}}", text, flags=re.IGNORECASE)
            
            if "EMAIL_ADDR" in seeds:
                text = text.replace(seeds["EMAIL_ADDR"], "{{EMAIL}}")

            for pii_type, pattern in self.patterns.items():
                text = re.sub(pattern, f"{{{{{pii_type}}}}}", text)

            scrubbed_data.append({
                "section": chunk['section'],
                "content": text,
                "metadata": chunk.get('metadata', {})
            })
            
        logger.info(f"Privacy scrubbing complete for {len(scrubbed_data)} chunks.")
        return scrubbed_data
import json
from pathlib import Path
from typing import Any, Dict, List

class RendererValidationError(ValueError):
    pass

class BaseRenderer:
    REQUIRED_FIELDS = [
        "professional_summary",
        "tailored_skills",
        "experience_bullets",
        "keyword_coverage",
        "notes",
    ]

    def validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise RendererValidationError("Payload must be a dictionary.")

        missing = [field for field in self.REQUIRED_FIELDS if field not in payload]
        if missing:
            raise RendererValidationError(f"Missing required fields: {', '.join(missing)}")

        payload = dict(payload)
        payload["professional_summary"] = str(payload.get("professional_summary", "")).strip()
        payload["tailored_skills"] = self._clean_str_list(payload.get("tailored_skills", []))
        payload["keyword_coverage"] = self._clean_str_list(payload.get("keyword_coverage", []))
        payload["notes"] = self._clean_str_list(payload.get("notes", []))
        payload["experience_bullets"] = self._clean_bullets(payload.get("experience_bullets", []))

        return payload

    def _clean_str_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        cleaned = []
        seen = set()
        for item in items:
            value = str(item).strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                cleaned.append(value)
        return cleaned

    def _clean_bullets(self, items: Any) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []
        cleaned = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            section = str(item.get("section", "Experience")).strip() or "Experience"
            original_evidence = str(item.get("original_evidence", "")).strip()
            tailored_bullet = str(item.get("tailored_bullet", "")).strip()
            if not tailored_bullet:
                continue
            signature = (section.lower(), tailored_bullet.lower())
            if signature in seen:
                continue
            seen.add(signature)
            cleaned.append({
                "section": section,
                "original_evidence": original_evidence,
                "tailored_bullet": tailored_bullet,
            })
        return cleaned

    def load_json_file(self, path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
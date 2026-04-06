from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import re


STANDARD_HEADINGS = {
    "summary", "professional summary", "experience", "work experience", "employment",
    "education", "skills", "technical skills", "projects", "certifications"
}

RISKY_PATTERNS = {
    "table_layout": [r"\|", r"\t{2,}"],
    "icons_or_glyphs": [r"[•◆▪■▶►★✓☑☎✉🔗]"],
    "nonstandard_headings": [r"career snapshot", r"value add", r"highlights profile"],
}

@dataclass
class RenderDecision:
    render_strategy: str
    ats_score: int
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    preserve_layout: bool = False
    normalize_headings: bool = False
    notes: List[str] = field(default_factory=list)


class ResumeRenderStrategy:
    def __init__(self):
        pass

    def decide(self, resume_text: str, extracted_structure: Dict[str, Any] | None = None) -> RenderDecision:
        if not resume_text or not resume_text.strip():
            return RenderDecision(
                render_strategy="rebuild_ats",
                ats_score=0,
                issues=["No resume text available for structure audit."],
                notes=["Fallback to ATS renderer because source resume structure could not be analyzed."]
            )

        text = resume_text.lower()
        score = 100
        issues: List[str] = []
        strengths: List[str] = []

        if self._looks_multicolumn_or_table(text):
            score -= 30
            issues.append("Resume may rely on multi-column or table-like layout.")
        else:
            strengths.append("Layout appears mostly linear.")

        heading_hits = self._count_standard_headings(text)
        if heading_hits >= 3:
            strengths.append("Standard section headings detected.")
        else:
            score -= 15
            issues.append("Too few standard ATS-friendly section headings detected.")

        if self._contains_risky_glyphs(resume_text):
            score -= 10
            issues.append("Decorative icons or glyphs detected that may affect ATS parsing.")

        if self._contains_nonstandard_headings(text):
            score -= 10
            issues.append("Nonstandard headings detected; heading normalization may help.")
        else:
            strengths.append("Headings appear conventional.")

        if extracted_structure:
            section_count = len(extracted_structure.get("sections", []))
            if section_count >= 3:
                strengths.append("Structured section extraction succeeded.")
            else:
                score -= 10
                issues.append("Section extraction was weak or incomplete.")

        score = max(0, min(100, score))

        if score >= 85:
            return RenderDecision(
                render_strategy="preserve_uploaded_format",
                ats_score=score,
                issues=issues,
                strengths=strengths,
                preserve_layout=True,
                normalize_headings=False,
                notes=["Uploaded resume appears ATS-safe enough to preserve its structure."]
            )

        if score >= 65:
            return RenderDecision(
                render_strategy="preserve_with_cleanup",
                ats_score=score,
                issues=issues,
                strengths=strengths,
                preserve_layout=True,
                normalize_headings=True,
                notes=["Preserve overall structure, but normalize headings and simplify risky formatting."]
            )

        return RenderDecision(
            render_strategy="rebuild_ats",
            ats_score=score,
            issues=issues,
            strengths=strengths,
            preserve_layout=False,
            normalize_headings=True,
            notes=["Resume structure appears risky for ATS parsing; rebuild using ATS template."]
        )

    def _looks_multicolumn_or_table(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in RISKY_PATTERNS["table_layout"])

    def _contains_risky_glyphs(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in RISKY_PATTERNS["icons_or_glyphs"])

    def _contains_nonstandard_headings(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in RISKY_PATTERNS["nonstandard_headings"])

    def _count_standard_headings(self, text: str) -> int:
        return sum(1 for heading in STANDARD_HEADINGS if heading in text)
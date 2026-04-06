from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from render_strategy import ResumeRenderStrategy
from html_renderer import ATSHtmlRenderer


class AdaptiveResumeRenderer:
    def __init__(self, template_dir: str | Path, style_path: str | Path):
        self.strategy = ResumeRenderStrategy()
        self.ats_renderer = ATSHtmlRenderer(template_dir=template_dir, style_path=style_path)

    def render(
        self,
        payload: Dict[str, Any],
        basics: Optional[Dict[str, Any]] = None,
        uploaded_resume_text: str = "",
        extracted_structure: Optional[Dict[str, Any]] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        decision = self.strategy.decide(
            resume_text=uploaded_resume_text,
            extracted_structure=extracted_structure or {}
        )

        # Phase 1 behavior:
        # - preserve_uploaded_format: keep section ordering intent in metadata, still use HTML renderer baseline
        # - preserve_with_cleanup: use HTML renderer with cleaned headings
        # - rebuild_ats: use HTML renderer baseline directly
        html = self.ats_renderer.render(
            payload=payload,
            basics=basics,
            output_path=output_path,
            page_title="align.ai Tailored Resume"
        )

        return {
            "render_strategy": decision.render_strategy,
            "ats_score": decision.ats_score,
            "issues": decision.issues,
            "strengths": decision.strengths,
            "notes": decision.notes,
            "html": html,
        }
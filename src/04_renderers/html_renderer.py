from pathlib import Path
from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

from base_renderer import BaseRenderer


class ATSHtmlRenderer(BaseRenderer):
    def __init__(self, template_dir: str | Path, style_path: str | Path):
        self.template_dir = Path(template_dir)
        self.style_path = Path(style_path)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(
        self,
        payload: Dict[str, Any],
        basics: Optional[Dict[str, Any]] = None,
        output_path: Optional[str | Path] = None,
        template_name: str = "ats_resume.html.j2",
        page_title: str = "Tailored Resume",
    ) -> str:
        data = self.validate_payload(payload)
        basics = basics or {}

        template = self.env.get_template(template_name)
        css = self.style_path.read_text(encoding="utf-8")

        grouped_experience = {}
        for bullet in data["experience_bullets"]:
            grouped_experience.setdefault(bullet["section"], []).append(bullet)

        html = template.render(
            page_title=page_title,
            basics=basics,
            css=css,
            summary=data["professional_summary"],
            skills=data["tailored_skills"],
            experience_groups=grouped_experience,
            keyword_coverage=data["keyword_coverage"],
            notes=data["notes"],
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding="utf-8")

        return html
import os
import json
import logging
from typing import List, Dict, Any

from google import genai
from google.genai import types


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.TailorAgent")


class TailorAgent:
    def __init__(self, model_id: str = "gemini-2.5-flash"):
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_id = model_id

        self.response_schema = {
            "type": "OBJECT",
            "required": [
                "professional_summary",
                "tailored_skills",
                "experience_bullets",
                "keyword_coverage",
                "notes"
            ],
            "properties": {
                "professional_summary": {
                    "type": "STRING",
                    "description": "A concise, resume-ready professional summary tailored to the job description."
                },
                "tailored_skills": {
                    "type": "ARRAY",
                    "description": "Skills safe to include in the tailored resume.",
                    "items": {
                        "type": "STRING"
                    }
                },
                "experience_bullets": {
                    "type": "ARRAY",
                    "description": "Grounded, resume-ready achievement bullets.",
                    "items": {
                        "type": "OBJECT",
                        "required": ["section", "original_evidence", "tailored_bullet"],
                        "properties": {
                            "section": {
                                "type": "STRING"
                            },
                            "original_evidence": {
                                "type": "STRING"
                            },
                            "tailored_bullet": {
                                "type": "STRING"
                            }
                        }
                    }
                },
                "keyword_coverage": {
                    "type": "ARRAY",
                    "description": "Job description keywords covered in the tailored output.",
                    "items": {
                        "type": "STRING"
                    }
                },
                "notes": {
                    "type": "ARRAY",
                    "description": "Optional notes about omitted or risky skills not included automatically.",
                    "items": {
                        "type": "STRING"
                    }
                }
            }
        }

    def _build_evidence_context(self, hybrid_results: List[Dict[str, Any]]) -> str:
        parts = []

        for idx, res in enumerate(hybrid_results, start=1):
            section = res.get("section") or res.get("metadata", {}).get("section", "GENERAL")
            content = res.get("content") or res.get("document", "")
            source = res.get("metadata", {}).get("source", "unknown_source")

            parts.append(
                f"[EVIDENCE_{idx}]\n"
                f"SECTION: {section}\n"
                f"SOURCE: {source}\n"
                f"CONTENT: {content}\n"
            )

        return "\n---\n".join(parts)

    def _post_process(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        parsed["professional_summary"] = (parsed.get("professional_summary") or "").strip()
        parsed["tailored_skills"] = parsed.get("tailored_skills", []) or []
        parsed["experience_bullets"] = parsed.get("experience_bullets", []) or []
        parsed["keyword_coverage"] = parsed.get("keyword_coverage", []) or []
        parsed["notes"] = parsed.get("notes", []) or []

        return parsed

    def generate_tailored_resume_content(
        self,
        job_description: str,
        hybrid_results: List[Dict[str, Any]],
        match_analysis: Dict[str, Any],
        approved_review_skills: List[str] | None = None,
    ) -> Dict[str, Any]:
        if not job_description or not job_description.strip():
            raise ValueError("job_description cannot be empty.")

        if not hybrid_results:
            raise ValueError("hybrid_results cannot be empty.")

        if not match_analysis:
            raise ValueError("match_analysis cannot be empty.")

        approved_review_skills = approved_review_skills or []

        evidence_context = self._build_evidence_context(hybrid_results)

        safe_to_apply = match_analysis.get("safe_to_apply", [])
        needs_user_review = match_analysis.get("needs_user_review", [])

        prompt = f"""
You are an expert resume tailoring writer for technical roles.

TASK:
Generate resume-ready tailored content using ONLY:
1. the provided resume evidence,
2. safe_to_apply skills from MatchAgent,
3. approved user-reviewed skills.

STRICT RULES:
1. Do not invent experience, projects, metrics, tools, or years.
2. Do not use any skill from needs_user_review unless it appears in approved_review_skills.
3. Keep the language professional, concise, and realistic.
4. Tailor the summary and bullets toward the job description.
5. Rewrite evidence into stronger wording, but remain grounded in the provided evidence.
6. If a skill is relevant but not approved, mention it only in notes, not in summary, skills, or bullets.
7. tailored_skills must include safe_to_apply skills and approved_review_skills only when grounded.
8. experience_bullets should be resume bullets, not paragraphs.

JOB DESCRIPTION:
{job_description}

SAFE_TO_APPLY:
{json.dumps(safe_to_apply, indent=2)}

NEEDS_USER_REVIEW:
{json.dumps(needs_user_review, indent=2)}

APPROVED_REVIEW_SKILLS:
{json.dumps(approved_review_skills, indent=2)}

RESUME EVIDENCE:
{evidence_context}
"""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.response_schema,
                temperature=0.2,
                candidate_count=1,
            ),
        )

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse TailorAgent JSON output: %s", e)
            raise ValueError(f"Invalid JSON returned by TailorAgent: {response.text}") from e

        parsed = self._post_process(parsed)

        logger.info(
            "Tailored content generated | skills=%s | bullets=%s | keywords=%s",
            len(parsed.get("tailored_skills", [])),
            len(parsed.get("experience_bullets", [])),
            len(parsed.get("keyword_coverage", [])),
        )

        return parsed


if __name__ == "__main__":
    sample_job_description = """
    Looking for a Data Engineer with experience in AWS, PySpark, Airflow,
    ETL pipelines, SQL, and PL/SQL. Experience with scalable data systems is preferred.
    """

    sample_results = [
        {
            "section": "EXPERIENCE: Flexon Technologies",
            "content": "Built real-time data streaming platform with Kafka and AWS Lambda.",
            "metadata": {"source": "resume.pdf"},
        },
        {
            "section": "SKILLS",
            "content": "Python, SQL, AWS, Kafka, ETL pipelines, Airflow, PySpark.",
            "metadata": {"source": "resume.pdf"},
        },
    ]

    sample_match_analysis = {
        "safe_to_apply": ["AWS", "PySpark", "Airflow", "ETL pipelines", "SQL"],
        "needs_user_review": [
            {
                "skill": "PL/SQL",
                "reason": "Adjacent to SQL but not explicitly proven.",
                "suggested_action": "Confirm before adding.",
                "review_type": "adjacent",
                "confidence": "MEDIUM"
            },
            {
                "skill": "Scalable data systems",
                "reason": "Strong related evidence exists from streaming platform work.",
                "suggested_action": "Confirm if you want to phrase it explicitly.",
                "review_type": "adjacent",
                "confidence": "HIGH"
            }
        ]
    }

    agent = TailorAgent()
    result = agent.generate_tailored_resume_content(
        job_description=sample_job_description,
        hybrid_results=sample_results,
        match_analysis=sample_match_analysis,
        approved_review_skills=["Scalable data systems"]
    )

    print(json.dumps(result, indent=2))
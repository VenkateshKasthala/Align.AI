import os
import json
import logging
from typing import List, Dict, Any


from google import genai
from google.genai import types


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlignAI.MatchAgent")


class MatchAgent:
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
            "description": "Structured job-description vs resume match analysis for technical roles.",
            "required": [
                "overall_match_score",
                "summary",
                "matched_skills",
                "adjacent_skills",
                "missing_skills",
                "safe_to_apply",
                "needs_user_review"
            ],
            "properties": {
                "overall_match_score": {
                    "type": "INTEGER",
                    "description": "Overall skill and evidence match score from 0 to 100."
                },
                "summary": {
                    "type": "STRING",
                    "description": "Short summary of the match analysis."
                },
                "matched_skills": {
                    "type": "ARRAY",
                    "description": "Skills directly supported by resume evidence.",
                    "items": {
                        "type": "OBJECT",
                        "required": ["skill", "evidence", "confidence"],
                        "properties": {
                            "skill": {
                                "type": "STRING",
                                "description": "JD skill explicitly matched in the resume evidence."
                            },
                            "evidence": {
                                "type": "ARRAY",
                                "description": "Specific evidence snippets supporting the match.",
                                "items": {
                                    "type": "STRING"
                                }
                            },
                            "confidence": {
                                "type": "STRING",
                                "description": "Confidence that this is a direct match.",
                                "enum": ["HIGH", "MEDIUM", "LOW"]
                            }
                        }
                    }
                },
                "adjacent_skills": {
                    "type": "ARRAY",
                    "description": "Skills not explicitly named but strongly related to existing resume evidence.",
                    "items": {
                        "type": "OBJECT",
                        "required": ["skill", "related_resume_signal", "reason", "confidence"],
                        "properties": {
                            "skill": {
                                "type": "STRING"
                            },
                            "related_resume_signal": {
                                "type": "STRING",
                                "description": "Resume phrase or signal that suggests adjacent capability."
                            },
                            "reason": {
                                "type": "STRING",
                                "description": "Why the skill is related but not exact."
                            },
                            "confidence": {
                                "type": "STRING",
                                "enum": ["HIGH", "MEDIUM", "LOW"]
                            }
                        }
                    }
                },
                "missing_skills": {
                    "type": "ARRAY",
                    "description": "Skills required by the JD that have no supporting evidence and no strong adjacent relationship.",
                    "items": {
                        "type": "OBJECT",
                        "required": ["skill", "reason"],
                        "properties": {
                            "skill": {
                                "type": "STRING"
                            },
                            "reason": {
                                "type": "STRING"
                            }
                        }
                    }
                },
                "safe_to_apply": {
                    "type": "ARRAY",
                    "description": "Skills safe for TailorAgent to use automatically without human confirmation.",
                    "items": {
                        "type": "STRING"
                    }
                },
                "needs_user_review": {
                    "type": "ARRAY",
                    "description": "Skills or concepts requiring human review before being used in tailoring.",
                    "items": {
                        "type": "OBJECT",
                        "required": [
                            "skill",
                            "reason",
                            "suggested_action",
                            "review_type",
                            "confidence"
                        ],
                        "properties": {
                            "skill": {
                                "type": "STRING"
                            },
                            "reason": {
                                "type": "STRING",
                                "description": "Why this item needs review."
                            },
                            "suggested_action": {
                                "type": "STRING",
                                "description": "What the user should do next."
                            },
                            "review_type": {
                                "type": "STRING",
                                "description": (
                                    "adjacent = related evidence exists; "
                                    "unfamiliar-but-plausible = not evidenced but user may know it from omitted/personal work; "
                                    "explicitly-missing = no evidence and no strong adjacent signal."
                                ),
                                "enum": [
                                    "adjacent",
                                    "unfamiliar-but-plausible",
                                    "explicitly-missing"
                                ]
                            },
                            "confidence": {
                                "type": "STRING",
                                "enum": ["HIGH", "MEDIUM", "LOW"]
                            }
                        }
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

    def _normalize_confidence(self, value: str) -> str:
        if not value:
            return "LOW"
        value = value.strip().upper()
        if value in {"HIGH", "MEDIUM", "LOW"}:
            return value
        return "LOW"

    def _dedupe_by_skill(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []

        for item in items:
            skill = (item.get("skill") or "").strip().lower()
            if not skill or skill in seen:
                continue
            seen.add(skill)
            deduped.append(item)

        return deduped

    def _post_process(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        parsed["overall_match_score"] = max(
            0,
            min(100, int(parsed.get("overall_match_score", 0)))
        )

        parsed["summary"] = (parsed.get("summary") or "").strip()

        for key in ["matched_skills", "adjacent_skills", "missing_skills", "needs_user_review"]:
            parsed[key] = parsed.get(key, []) or []

        parsed["safe_to_apply"] = parsed.get("safe_to_apply", []) or []

        for item in parsed["matched_skills"]:
            item["confidence"] = self._normalize_confidence(item.get("confidence"))

        for item in parsed["adjacent_skills"]:
            item["confidence"] = self._normalize_confidence(item.get("confidence"))

        for item in parsed["needs_user_review"]:
            item["confidence"] = self._normalize_confidence(item.get("confidence"))
            review_type = (item.get("review_type") or "").strip().lower()
            if review_type not in {
                "adjacent",
                "unfamiliar-but-plausible",
                "explicitly-missing"
            }:
                item["review_type"] = "explicitly-missing"
            else:
                item["review_type"] = review_type

        parsed["matched_skills"] = self._dedupe_by_skill(parsed["matched_skills"])
        parsed["adjacent_skills"] = self._dedupe_by_skill(parsed["adjacent_skills"])
        parsed["missing_skills"] = self._dedupe_by_skill(parsed["missing_skills"])
        parsed["needs_user_review"] = self._dedupe_by_skill(parsed["needs_user_review"])

        safe_lower = {s.strip().lower() for s in parsed["safe_to_apply"] if s.strip()}
        matched_lower = {
            (item.get("skill") or "").strip().lower()
            for item in parsed["matched_skills"]
            if (item.get("skill") or "").strip()
        }

        parsed["safe_to_apply"] = [
            item.get("skill")
            for item in parsed["matched_skills"]
            if (item.get("skill") or "").strip().lower() in matched_lower
        ]

        return parsed

    def analyze_match(
        self,
        job_description: str,
        hybrid_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not job_description or not job_description.strip():
            raise ValueError("job_description cannot be empty.")

        if not hybrid_results:
            raise ValueError("hybrid_results cannot be empty.")

        evidence_context = self._build_evidence_context(hybrid_results)

        prompt = f"""
You are an expert resume-job matching analyst for technical roles.

TASK:
Compare the job description with the provided resume evidence and return a structured JSON analysis.

STRICT CLASSIFICATION RULES:
1. Only use the provided resume evidence.
2. matched_skills = direct, explicit evidence exists in the resume.
3. adjacent_skills = the skill is not explicitly stated, but there is strong related evidence.
   Example: SQL may support PL/SQL as adjacent, but not as a direct match.
4. missing_skills = no evidence and no strong adjacent relationship exists.
5. safe_to_apply = only direct matched skills that TailorAgent can safely use automatically.
6. needs_user_review must include only items that require user confirmation before use.
7. review_type meanings:
   - adjacent: related evidence exists in the resume, but the exact skill is not explicitly proven.
   - unfamiliar-but-plausible: not evidenced in the resume, but a reasonable user may know it from omitted work, coursework, certifications, or personal projects.
   - explicitly-missing: no evidence and no strong adjacent signal.
8. If there is strong related evidence, prefer review_type = adjacent, not unfamiliar-but-plausible.
9. Do not invent tools, projects, years of experience, or achievements.
10. overall_match_score must be an integer from 0 to 100.
11. confidence must be one of HIGH, MEDIUM, or LOW.
12. Keep the output conservative and interview-safe.

JOB DESCRIPTION:
{job_description}

RESUME EVIDENCE:
{evidence_context}
"""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.response_schema,
                temperature=0.0,
                candidate_count=1,
            ),
        )

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse MatchAgent JSON output: %s", e)
            raise ValueError(f"Invalid JSON returned by MatchAgent: {response.text}") from e

        parsed = self._post_process(parsed)
        
        logger.info(
            "Match analysis completed with score=%s | matched=%s | adjacent=%s | missing=%s | review=%s",
            parsed.get("overall_match_score"),
            len(parsed.get("matched_skills", [])),
            len(parsed.get("adjacent_skills", [])),
            len(parsed.get("missing_skills", [])),
            len(parsed.get("needs_user_review", [])),
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

    agent = MatchAgent()
    result = agent.analyze_match(sample_job_description, sample_results)
    print(json.dumps(result, indent=2))
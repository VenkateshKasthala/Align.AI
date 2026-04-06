import json
from pathlib import Path

from html_renderer import ATSHtmlRenderer


SAMPLE_BASICS = {
    "name": "Venkatesh Kastala",
    "email": "venkateshkastala0803@gmail.com",
    "phone": "(201) 241-2979",
    "linkedin": "linkedin.com/in/venkatesh-k-972133328",
    "location": "West Chester, PA"
}

SAMPLE_PAYLOAD = {
    "professional_summary": "Experienced Data Engineer with a strong background in designing and implementing robust ETL pipelines, leveraging AWS, PySpark, and Airflow. Proven ability to build scalable data systems, including real-time streaming platforms, and proficient in SQL.",
    "tailored_skills": [
        "AWS",
        "PySpark",
        "Airflow",
        "ETL pipelines",
        "SQL",
        "Scalable data systems"
    ],
    "experience_bullets": [
        {
            "section": "EXPERIENCE: Flexon Technologies",
            "original_evidence": "Built real-time data streaming platform with Kafka and AWS Lambda.",
            "tailored_bullet": "Engineered and deployed a real-time data streaming platform using Kafka and AWS Lambda to support scalable data processing workflows."
        }
    ],
    "keyword_coverage": [
        "AWS",
        "PySpark",
        "Airflow",
        "ETL pipelines",
        "SQL",
        "Scalable data systems"
    ],
    "notes": [
        "PL/SQL was requested in the job description but was not added because it was not explicitly proven in the resume evidence."
    ]
}


def main():
    root = Path(__file__).parent
    renderer = ATSHtmlRenderer(
        template_dir=root / "templates",
        style_path=root / "styles" / "ats_resume.css",
    )

    output_file = root / "tailored_resume.html"
    renderer.render(
        payload=SAMPLE_PAYLOAD,
        basics=SAMPLE_BASICS,
        output_path=output_file,
        page_title="align.ai Tailored Resume",
    )

    print(json.dumps({"output_html": str(output_file)}, indent=2))


if __name__ == "__main__":
    main()
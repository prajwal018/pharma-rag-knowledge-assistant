
import re
from openai import OpenAI
from app.config import MISTRAL_API_KEY, MISTRAL_MODEL

SYSTEM_PROMPT = """
You are an evidence-grounded pharmaceutical document assistant.

STRICT CITATION RULES:

1. Answer only using the supplied evidence.
2. Every factual claim must have a citation.
3. Cite evidence using only the IDs provided:
   [S1], [S2], [S3], etc.
4. NEVER generate PDF page numbers yourself.
5. NEVER invent source names or citation IDs.
6. If multiple passages support an answer, cite all relevant IDs.
7. If evidence is insufficient, explicitly say so.
8. Distinguish historical decisions from current status.
9. Do not make medical or clinical recommendations.

IMPORTANT:
The Python application will convert your evidence IDs
into actual document names and page numbers.

Return a clear, structured answer.
"""

def build_context(results: list[dict]):
    context_parts = []
    citation_map = {}
    for i, item in enumerate(results, start=1):
        evidence_id = f"S{i}"
        citation_map[evidence_id] = {
            "source": item["source"],
            "page": item["page"],
            "score": item.get("score", 0)
        }
        context_parts.append(
            f"""
            [{evidence_id}]
            Source: {item['source']}
            Page: {item['page']}

            Evidence:
            {item['text']}
            """
        )
    return "\n\n".join(context_parts), citation_map


def validate_citations(answer: str, citation_map: dict):
    valid_ids = set(citation_map.keys())
    found_ids = re.findall(r"\[(S\d+)\]", answer)
    used_ids = list(dict.fromkeys(
        evidence_id
        for evidence_id in found_ids
        if evidence_id in valid_ids
    ))

    # Remove invalid citation IDs
    answer = re.sub(
        r"\[(S\d+)\]",
        lambda match: (
            match.group(0)
            if match.group(1) in valid_ids
            else ""
        ),
        answer
    )

    # Convert valid evidence IDs into actual source references
    for evidence_id in used_ids:
        source = citation_map[evidence_id]
        reference = (
            f"[Source: {source['source']}, "
            f"Page {source['page']}]"
        )
        answer = answer.replace(
            f"[{evidence_id}]",
            reference
        )

    if not used_ids:
        answer += (
            "\n\nCitation validation warning: "
            "The model did not provide a valid evidence citation. "
            "Please inspect the retrieved passages below."
        )
    return answer, used_ids


def answer_question(question: str, results: list[dict]):
    if not results:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    if not MISTRAL_API_KEY:
        return "MISTRAL_API_KEY is not configured."

    context, citation_map = build_context(results)
    try:
        client = OpenAI(
            api_key=MISTRAL_API_KEY,
            base_url="https://api.mistral.ai/openai/v1"
        )
        response = client.chat.completions.create(
            model=MISTRAL_MODEL,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": f"""
                    Question:
                    {question}
                    Retrieved evidence:
                    {context}
                    Answer the question using the evidence.
                    Cite every factual claim using the
                    provided evidence IDs only.
                    Example:
                    DV-21 closed on 18 September [S1].
                    Do not generate page numbers.
                    """
                }
            ]
        )
        raw_answer = (
            response.choices[0].message.content
            or "No answer was generated."
        )
        final_answer, used_ids = validate_citations(
            raw_answer,
            citation_map
        )
        return final_answer
    except Exception as exc:
        return f"Mistral API Error: {str(exc)}"


"""Optional LLM-backed annotation helpers.

These are convenience utilities for *annotating* analysis results with a large
language model (e.g. summarising which CNV genes on an arm are notable in a
given cancer type). They are deliberately kept out of the numeric core
(``core/Clustering.py``): the LLM client is **injected by the caller**, so
``pymaftools`` takes on no hard dependency on ``openai``.
"""

from __future__ import annotations


def gpt_known_genes_summary(
    client: object,
    genes: list[str],
    arm: str,
    cancer_type: str = "lung cancer",
) -> tuple[str, str]:
    """
    Query GPT-4 for well-known genes in a given chromosomal arm and cancer type.

    Parameters
    ----------
    client : object
        OpenAI client instance with a ``chat.completions.create`` method.
    genes : list[str]
        List of gene names to evaluate.
    arm : str
        Chromosomal arm where the genes are located (e.g., ``'3p'``).
    cancer_type : str, optional
        Cancer type context for the query, by default ``'lung cancer'``.

    Returns
    -------
    str
        GPT-4 response text listing notable genes and reasons.
    str
        The prompt that was sent to the model.
    """
    prompt = "\n".join(
        [
            f"The following is a list of human CNV genes located on {arm}:",
            ", ".join(genes),
            "",
            f"Based on cancer literature, known functions, and biomedical research value, identify the well-known and frequently studied genes among these in {cancer_type}, and briefly explain why.",
            "Use the following format:",
            "```",
            "Gene: gene_name, Reason: brief explanation",
            "```",
            "Do not add numbers or dashes. Do not use multiple lines or paragraphs for explanation. Output one gene per line.",
        ]
    )

    result = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        result = response.choices[0].message.content
    except Exception as e:
        print("Error:", e)

    return result, prompt

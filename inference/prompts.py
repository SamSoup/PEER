from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class PromptTemplate:
    system: str
    example_header: str
    pair_intro: str
    score_instruction: str
    example_formatter: Callable[[dict, int], str]
    cot_suffix: str = (
        "Let's think step by step. After your reasoning, provide ONLY the final numeric score."
    )

    def build_user_message(
        self,
        sentence_a: str,
        sentence_b: str,
        icl_examples: List[dict],
        use_cot: bool,
    ) -> str:
        parts: List[str] = []
        if icl_examples:
            parts.append(self.example_header)
            for i, ex in enumerate(icl_examples, start=1):
                parts.append(self.example_formatter(ex, i))
            parts.append("\nNow score the new pair.\n")
        parts.append(
            self.pair_intro.format(
                sentence_a=sentence_a.strip(), sentence_b=sentence_b.strip()
            )
        )
        parts.append(self.score_instruction)
        if use_cot:
            parts.append(self.cot_suffix)
        return "\n".join(parts)


def _similarity_example(ex: dict, idx: int, label: str) -> str:
    return (
        f"Example {idx}:\n"
        f"Sentence A: {ex['sentence1']}\n"
        f"Sentence B: {ex['sentence2']}\n"
        f"{label}: {ex['score_text']}\n"
    )


PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "stsb": PromptTemplate(
        system=(
            "You are an expert human annotator for semantic textual similarity (STS).\n"
            "You will see English sentence pairs and you must judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 0.000 to 1.000.\n"
            "- 1.000 = same meaning / paraphrases.\n"
            "- 0.500 = partially similar meaning but with important differences or missing info.\n"
            "- 0.000 = completely unrelated meaning.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction=(
            "Answer with ONLY the similarity score in the range 0.000 to 1.000, "
            "using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: _similarity_example(
            ex, idx, label="Similarity"
        ),
    ),
    "stsbenchmark_mteb": PromptTemplate(
        system=(
            "You are an expert human annotator for semantic textual similarity (STS).\n"
            "You will see English sentence pairs and judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 0.000 to 5.000, inclusive.\n"
            "- 5.000 = same meaning / paraphrases.\n"
            "- 2.500 ≈ partially similar with important differences.\n"
            "- 0.000 = completely unrelated meaning.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction=(
            "Answer with ONLY the similarity score in the range 0.000..5.000 (inclusive), "
            "using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: _similarity_example(
            ex, idx, label="Similarity (0..5)"
        ),
    ),
    "sickr_sts": PromptTemplate(
        system=(
            "You are an expert annotator for semantic textual similarity (STS) for Sentences Involving Compositional Knowldedge.\n"
            "You will see sentence pairs and judge how similar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 1.000 to 5.000, inclusive.\n"
            "- 5.000 = same meaning / paraphrases; 1.000 = completely unrelated.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header="Here are some examples of how to score similarity:\n",
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction=(
            "Answer with ONLY the similarity score in the range 1.000..5.000 (inclusive), "
            "using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: _similarity_example(
            ex, idx, label="Similarity (1..5)"
        ),
    ),
    "sts22": PromptTemplate(
        system=(
            "You are an expert annotator for semantic textual similarity (STS).\n"
            "You will see news article pairs that may be in DIFFERENT languages; judge how dissimilar in MEANING they are.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued similarity score from 1.000 to 4.000, inclusive.\n"
            "- 4.000 = NO similarity at all; 1.000 = two stories are almost identical.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score, with EXACTLY three digits after the decimal point (e.g., 3.000).\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header=(
            "Here are some examples of how to score similarity (pairs may be cross-lingual):\n"
        ),
        pair_intro="Sentence A: {sentence_a}\nSentence B: {sentence_b}\n\n",
        score_instruction=(
            "Answer with ONLY the similarity score in the range 1.000..4.000 (inclusive), "
            "using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: _similarity_example(
            ex, idx, label="Similarity (1..4)"
        ),
    ),
    "wmt_en_ru": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in Russian (source) and Sentence B is in English (translation).\n"
            "You must judge how well the meaning is preserved from Russian to English.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header=(
            "Here are some examples of how to score Russian (source) to English (translation) quality / similarity:\n"
        ),
        pair_intro=(
            "Sentence A (Russian source): {sentence_a}\n"
            "Sentence B (English translation): {sentence_b}\n\n"
        ),
        score_instruction=(
            "Answer with ONLY the translation quality / similarity score in the range "
            "0.000 to 100.000, using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: (
            f"Example {idx}:\n"
            f"Sentence A (Russian source): {ex['sentence1']}\n"
            f"Sentence B (English translation): {ex['sentence2']}\n"
            f"Score (0-100): {ex['score_text']}\n"
        ),
    ),
    "wmt_en_zh": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in English (source) and Sentence B is in Chinese (translation).\n"
            "You must judge how well the meaning is preserved from English to Chinese.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header=(
            "Here are some examples of how to score English (source) to Chinese (translation) quality / similarity:\n"
        ),
        pair_intro=(
            "Sentence A (English source): {sentence_a}\n"
            "Sentence B (Chinese translation): {sentence_b}\n\n"
        ),
        score_instruction=(
            "Answer with ONLY the translation quality / similarity score in the range "
            "0.000 to 100.000, using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: (
            f"Example {idx}:\n"
            f"Sentence A (English source): {ex['sentence1']}\n"
            f"Sentence B (Chinese translation): {ex['sentence2']}\n"
            f"Score (0-100): {ex['score_text']}\n"
        ),
    ),
    "wmt_si_en": PromptTemplate(
        system=(
            "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
            "You will see sentence pairs where Sentence A is in Sinhala (source) and Sentence B is in English (translation).\n"
            "You must judge how well the meaning is preserved from Sinhala to English.\n\n"
            "Scoring instructions:\n"
            "- Output a real-valued translation quality / similarity score from 0.000 to 100.000.\n"
            "- 100.000 = perfect meaning preservation (excellent translation / paraphrases).\n"
            "- 50.000  = partially similar meaning but with important differences or missing info.\n"
            "- 0.000   = completely unrelated or very bad translation.\n\n"
            "CRITICAL FORMAT RULE:\n"
            "Return ONLY the numeric score with EXACTLY three digits after the decimal point.\n"
            "Do NOT include any words, punctuation, units, labels, or explanation.\n"
        ),
        example_header=(
            "Here are some examples of how to score Sinhala (source) to English (translation) quality / similarity:\n"
        ),
        pair_intro=(
            "Sentence A (Sinhala source): {sentence_a}\n"
            "Sentence B (English translation): {sentence_b}\n\n"
        ),
        score_instruction=(
            "Answer with ONLY the translation quality / similarity score in the range "
            "0.000 to 100.000, using exactly three digits after the decimal point:"
        ),
        example_formatter=lambda ex, idx: (
            f"Example {idx}:\n"
            f"Sentence A (Sinhala source): {ex['sentence1']}\n"
            f"Sentence B (English translation): {ex['sentence2']}\n"
            f"Score (0-100): {ex['score_text']}\n"
        ),
    ),
}

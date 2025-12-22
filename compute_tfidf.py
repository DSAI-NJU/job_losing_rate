from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_policy_frequency import DATA_PATH, KEYWORDS, filter_by_keywords, load_data

OUTPUT_TOP_TERMS = Path(__file__).parent / "employment_policy_tfidf_top_terms.png"
OUTPUT_TABLE = Path(__file__).parent / "employment_policy_tfidf.csv"


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> list[str]:
    """Extract Chinese word-like tokens (>=2 characters)."""
    return TOKEN_PATTERN.findall(text)


def compute_tfidf(corpus: list[str]) -> tuple[list[str], list[float]]:
    """Compute TF-IDF mean score for each term in the corpus."""
    tokenized = [tokenize(doc) for doc in corpus]
    n_docs = len(tokenized)
    if n_docs == 0:
        return [], []

    doc_freq: Counter[str] = Counter()
    term_freqs: list[tuple[Counter[str], int]] = []

    for tokens in tokenized:
        counts = Counter(tokens)
        term_freqs.append((counts, sum(counts.values())))
        doc_freq.update(set(tokens))

    aggregated: defaultdict[str, float] = defaultdict(float)

    for counts, total_tokens in term_freqs:
        if total_tokens == 0:
            continue
        for term, count in counts.items():
            tf = count / total_tokens
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
            aggregated[term] += tf * idf

    if not aggregated:
        return [], []

    ordered = sorted(aggregated.items(), key=lambda item: item[1] / n_docs, reverse=True)
    terms = [term for term, _ in ordered]
    scores = [value / n_docs for _, value in ordered]
    return terms, scores


def build_monthly_tfidf(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Aggregate top TF-IDF terms per month."""
    records: list[dict[str, str | float]] = []
    for period, group in df.groupby(df["publish_date"].dt.to_period("M")):
        corpus = group["title"].dropna().astype(str).tolist()
        if not corpus:
            continue
        terms, scores = compute_tfidf(corpus)
        for term, score in zip(terms[:top_n], scores[:top_n]):
            records.append({"month": str(period), "term": term, "tfidf": float(score)})
    return pd.DataFrame(records)


def plot_top_terms(overall_terms: list[str], overall_scores: list[float], top_n: int = 20) -> None:
    """Plot the top N TF-IDF terms ranked by mean score."""
    plt.rcParams["font.sans-serif"] = [
        "Heiti TC",
        "PingFang HK",
        "PingFang SC",
        "Songti SC",
        "Arial Unicode MS",
        plt.rcParams.get("font.sans-serif", ["DejaVu Sans"])[0],
    ]
    plt.rcParams["axes.unicode_minus"] = False

    top_terms = overall_terms[:top_n]
    top_scores = overall_scores[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(top_terms)), top_scores)
    ax.set_xticks(range(len(top_terms)))
    ax.set_xticklabels(top_terms, rotation=45, ha="right")
    ax.set_ylabel("平均 TF-IDF")
    ax.set_title("2024-2025 年涉就业政策标题高频关键词 (TF-IDF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_TOP_TERMS, dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_data(DATA_PATH)
    filtered = filter_by_keywords(df, KEYWORDS)
    if filtered.empty:
        print("没有匹配政策，无法计算 TF-IDF。")
        return

    corpus = filtered["title"].dropna().astype(str).tolist()
    overall_terms, overall_scores = compute_tfidf(corpus)
    monthly_tfidf = build_monthly_tfidf(filtered, top_n=10)

    monthly_tfidf.to_csv(OUTPUT_TABLE, index=False)
    plot_top_terms(overall_terms, overall_scores, top_n=20)

    print("总体 TF-IDF 前 20:")
    for term, score in zip(overall_terms[:20], overall_scores[:20]):
        print(f"{term}\t{score:.4f}")

    print(f"月度 TF-IDF 已保存: {OUTPUT_TABLE}")
    print(f"TF-IDF 图表已保存: {OUTPUT_TOP_TERMS}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).parent / "policy_titles_2024_2025.csv"
OUTPUT_PATH = Path(__file__).parent / "employment_policy_frequency.png"
KEYWORDS: Sequence[str] = ("就业", "社保", "失业", "劳动")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load policy data with parsed dates."""
    df = pd.read_csv(csv_path, dtype={"title": "string"})
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%Y-%m-%d", errors="coerce")
    return df.dropna(subset=["publish_date", "title"])


def filter_by_keywords(df: pd.DataFrame, keywords: Sequence[str]) -> pd.DataFrame:
    """Return rows whose title contains any keyword (case-sensitive)."""
    pattern = "|".join(keywords)
    mask = df["title"].str.contains(pattern, regex=True, na=False)
    return df.loc[mask].copy()


def monthly_counts(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Aggregate counts by month between start and end inclusive."""
    df = df[(df["publish_date"] >= start) & (df["publish_date"] <= end)]
    if df.empty:
        return pd.DataFrame({"month": pd.period_range(start, end, freq="M"), "count": 0})

    counts = (
        df.assign(month=df["publish_date"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    full_range = pd.period_range(start, end, freq="M")
    counts = counts.set_index("month").reindex(full_range, fill_value=0).reset_index()
    counts.rename(columns={"index": "month"}, inplace=True)
    return counts


def plot_counts(counts: pd.DataFrame, output_path: Path) -> None:
    """Plot monthly counts and save to disk."""
    # Try to use a font that supports Chinese characters.
    plt.rcParams["font.sans-serif"] = [
        "Heiti TC",
        "PingFang HK",
        "PingFang SC",
        "Songti SC",
        "Arial Unicode MS",
        plt.rcParams.get("font.sans-serif", ["DejaVu Sans"])[0],
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(counts["month"].astype(str), counts["count"], marker="o", linewidth=2)
    ax.set_xlabel("月份")
    ax.set_ylabel("政策数量")
    ax.set_title("2024-2025年涉就业政策月度数量")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    df = load_data(DATA_PATH)
    filtered = filter_by_keywords(df, KEYWORDS)
    counts = monthly_counts(filtered, start="2024-01-01", end="2025-12-31")
    plot_counts(counts, OUTPUT_PATH)
    counts["month"] = counts["month"].astype(str)
    counts.to_csv(Path(__file__).parent / "employment_policy_frequency.csv", index=False)
    print(counts.to_string(index=False))
    print(f"图表已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

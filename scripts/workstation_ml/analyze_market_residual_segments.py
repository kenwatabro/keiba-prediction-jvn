import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "datasets" / "train_data.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "evaluations" / "experiments" / "market_residual_segments_summary.json"
)


@dataclass(frozen=True)
class CandidateSpec:
    mode: str
    segment_columns: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.mode}:{'+'.join(self.segment_columns)}"


SINGLE_SEGMENTS = [
    "JyoCD",
    "TrackCD",
    "DistanceBucket",
    "GradeCD",
    "JyuryoCD",
    "SexCD",
    "BareiBand",
    "FieldSizeBand",
    "OddsBand",
    "NinkiBand",
    "HorseStartsBeforeBand",
    "HorseDaysSinceLastRaceBand",
    "HorseDistanceChangeBand",
    "HorseLast3Top3RateBand",
    "JockeyWinRateSmoothBand",
    "TrainerWinRateSmoothBand",
]
INTERACTION_SEGMENTS = [
    ("DistanceBucket", "OddsBand"),
    ("TrackCD", "OddsBand"),
    ("JyoCD", "OddsBand"),
    ("FieldSizeBand", "OddsBand"),
    ("HorseDaysSinceLastRaceBand", "OddsBand"),
    ("HorseDistanceChangeBand", "OddsBand"),
    ("JockeyWinRateSmoothBand", "OddsBand"),
    ("TrainerWinRateSmoothBand", "OddsBand"),
    ("DistanceBucket", "NinkiBand"),
    ("TrackCD", "NinkiBand"),
    ("FieldSizeBand", "NinkiBand"),
]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _safe_qcut(series: pd.Series, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().nunique() < 2:
        return pd.Series("missing", index=series.index, dtype="string")
    try:
        return (
            pd.qcut(numeric, q=len(labels), labels=labels, duplicates="drop")
            .astype("string")
            .fillna("missing")
        )
    except ValueError:
        return pd.Series("missing", index=series.index, dtype="string")


def _cut_numeric(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (
        pd.cut(numeric, bins=bins, labels=labels, include_lowest=True, right=False)
        .astype("string")
        .fillna("missing")
    )


def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["RaceDate"] = pd.to_datetime(result["RaceDate"], errors="coerce")
    result["OddsDecimal"] = _coerce_numeric(result["OddsDecimal"], fill_value=0.0)
    result["Ninki"] = _coerce_numeric(result["Ninki"], fill_value=999.0)
    result["TargetWin"] = _coerce_numeric(result["TargetWin"], fill_value=0.0).astype(int)
    result = result.loc[result["RaceDate"].notna() & result["OddsDecimal"].gt(0)].copy()

    result["GrossReturn"] = result["TargetWin"] * result["OddsDecimal"] * 100.0
    result["RaceFieldSize"] = result.groupby("RaceKey")["RaceKey"].transform("size")
    implied = 1.0 / result["OddsDecimal"]
    implied_sum = implied.groupby(result["RaceKey"]).transform("sum")
    result["MarketProbNorm"] = (implied / implied_sum).fillna(0.0)
    result["MarketResidual"] = result["TargetWin"] - result["MarketProbNorm"]

    result["OddsBand"] = _cut_numeric(
        result["OddsDecimal"],
        [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 9999.0],
        ["1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20-50", "50plus"],
    )
    result["NinkiBand"] = _cut_numeric(
        result["Ninki"],
        [1.0, 2.0, 4.0, 6.0, 10.0, 18.0, 999.0],
        ["1", "2-3", "4-5", "6-9", "10-17", "18plus"],
    )
    result["BareiBand"] = _cut_numeric(
        result.get("Barei", pd.Series(index=result.index)),
        [0.0, 3.0, 4.0, 5.0, 7.0, 99.0],
        ["2under", "3", "4", "5-6", "7plus"],
    )
    result["FieldSizeBand"] = _cut_numeric(
        result["RaceFieldSize"],
        [0.0, 8.0, 12.0, 15.0, 19.0, 99.0],
        ["7under", "8-11", "12-14", "15-18", "19plus"],
    )
    result["HorseStartsBeforeBand"] = _cut_numeric(
        result.get("HorseStartsBefore", pd.Series(index=result.index)),
        [0.0, 1.0, 4.0, 8.0, 16.0, 999.0],
        ["0", "1-3", "4-7", "8-15", "16plus"],
    )
    result["HorseDaysSinceLastRaceBand"] = _cut_numeric(
        result.get("HorseDaysSinceLastRace", pd.Series(index=result.index)),
        [-9999.0, 1.0, 28.0, 56.0, 112.0, 365.0, 9999.0],
        ["missing_or_0", "1-27", "28-55", "56-111", "112-364", "365plus"],
    )
    result["HorseDistanceChangeBand"] = _cut_numeric(
        result.get("HorseDistanceChange", pd.Series(index=result.index)),
        [-9999.0, -600.0, -200.0, 200.0, 600.0, 9999.0],
        ["shorten_600plus", "shorten_200_599", "sameish", "extend_200_599", "extend_600plus"],
    )
    result["HorseLast3Top3RateBand"] = _safe_qcut(
        result.get("HorseLast3Top3Rate", pd.Series(index=result.index)),
        ["q1_low", "q2", "q3", "q4_high"],
    )
    result["JockeyWinRateSmoothBand"] = _safe_qcut(
        result.get("JockeyWinRateSmoothBefore", pd.Series(index=result.index)),
        ["q1_low", "q2", "q3", "q4_high"],
    )
    result["TrainerWinRateSmoothBand"] = _safe_qcut(
        result.get("TrainerWinRateSmoothBefore", pd.Series(index=result.index)),
        ["q1_low", "q2", "q3", "q4_high"],
    )
    return result


def select_period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["RaceDate"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def select_favorite_picks(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal", "Umaban"], kind="stable")
        .groupby("RaceKey", as_index=False)
        .first()
    )


def summarize_picks(picks: pd.DataFrame) -> dict[str, float | int]:
    bets = len(picks)
    stake = bets * 100.0
    gross_return = float(_coerce_numeric(picks.get("GrossReturn", pd.Series(dtype=float))).sum())
    market_prob = _coerce_numeric(picks.get("MarketProbNorm", pd.Series(dtype=float))).mean() if bets else 0.0
    hit_rate = _coerce_numeric(picks.get("TargetWin", pd.Series(dtype=float))).mean() if bets else 0.0
    return {
        "bets": int(bets),
        "hit_rate": float(hit_rate),
        "return_rate": float(gross_return / stake * 100.0) if stake else 0.0,
        "mean_market_prob": float(market_prob),
        "win_minus_market_prob": float(hit_rate - market_prob),
        "mean_odds": float(_coerce_numeric(picks.get("OddsDecimal", pd.Series(dtype=float))).mean()) if bets else 0.0,
    }


def summarize_yearly(picks: pd.DataFrame) -> list[dict[str, float | int]]:
    if picks.empty:
        return []
    return [
        {"year": int(year), **summarize_picks(year_picks)}
        for year, year_picks in picks.groupby(picks["RaceDate"].dt.year, sort=True)
    ]


def _segment_value(row: pd.Series, columns: tuple[str, ...]) -> str:
    return " | ".join(f"{column}={row[column]}" for column in columns)


def evaluate_candidate(
    validation_source: pd.DataFrame,
    test_source: pd.DataFrame,
    full_source: pd.DataFrame,
    spec: CandidateSpec,
    min_validation_bets: int,
) -> list[dict[str, object]]:
    missing_columns = [column for column in spec.segment_columns if column not in validation_source.columns]
    if missing_columns:
        return []

    rows: list[dict[str, object]] = []
    grouped = validation_source.groupby(list(spec.segment_columns), dropna=False, sort=True)
    for key, validation_picks in grouped:
        if len(validation_picks) < min_validation_bets:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        mask = pd.Series(True, index=test_source.index)
        for column, value in zip(spec.segment_columns, key, strict=True):
            mask &= test_source[column].eq(value)
        test_picks = test_source.loc[mask].copy()
        full_mask = pd.Series(True, index=full_source.index)
        for column, value in zip(spec.segment_columns, key, strict=True):
            full_mask &= full_source[column].eq(value)
        full_picks = full_source.loc[full_mask].copy()
        value_row = pd.Series(dict(zip(spec.segment_columns, key, strict=True)))
        rows.append(
            {
                "candidate": spec.name,
                "mode": spec.mode,
                "segment_columns": list(spec.segment_columns),
                "segment_value": _segment_value(value_row, spec.segment_columns),
                "validation": summarize_picks(validation_picks),
                "test": summarize_picks(test_picks),
                "test_yearly": summarize_yearly(test_picks),
                "full_period": summarize_picks(full_picks),
                "full_yearly": summarize_yearly(full_picks),
            }
        )
    return rows


def build_candidate_specs() -> list[CandidateSpec]:
    specs = [CandidateSpec("favorite", (segment,)) for segment in SINGLE_SEGMENTS]
    specs.extend(CandidateSpec("all_runners", (segment,)) for segment in SINGLE_SEGMENTS)
    specs.extend(CandidateSpec("all_runners", tuple(segment_pair)) for segment_pair in INTERACTION_SEGMENTS)
    return specs


def analyze_segments(
    frame: pd.DataFrame,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    min_validation_bets: int,
    top_n: int,
) -> dict[str, object]:
    prepared = add_analysis_columns(frame)
    validation = select_period(prepared, validation_start, validation_end)
    test = select_period(prepared, test_start, test_end)
    validation_favorites = select_favorite_picks(validation)
    test_favorites = select_favorite_picks(test)
    full_favorites = select_favorite_picks(prepared)
    sources = {
        "favorite": (validation_favorites, test_favorites, full_favorites),
        "all_runners": (validation, test, prepared),
    }

    candidates: list[dict[str, object]] = []
    for spec in build_candidate_specs():
        validation_source, test_source, full_source = sources[spec.mode]
        candidates.extend(evaluate_candidate(validation_source, test_source, full_source, spec, min_validation_bets))
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["validation"]["return_rate"],  # type: ignore[index]
            row["validation"]["bets"],  # type: ignore[index]
        ),
        reverse=True,
    )

    return {
        "validation_period": [validation_start, validation_end],
        "test_period": [test_start, test_end],
        "min_validation_bets": min_validation_bets,
        "top_n": top_n,
        "deployability": "research_final_odds_only",
        "baselines": {
            "full_favorite": summarize_picks(full_favorites),
            "validation_favorite": summarize_picks(validation_favorites),
            "test_favorite": summarize_picks(test_favorites),
            "full_all_runners": summarize_picks(prepared),
            "validation_all_runners": summarize_picks(validation),
            "test_all_runners": summarize_picks(test),
        },
        "top_validation_segments": candidates[:top_n],
        "top_test_segments_among_validation_selected": sorted(
            candidates[: max(top_n * 3, top_n)],
            key=lambda row: (
                row["test"]["return_rate"],  # type: ignore[index]
                row["test"]["bets"],  # type: ignore[index]
            ),
            reverse=True,
        )[:top_n],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze final-odds market residuals by race and horse segments.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-06-21")
    parser.add_argument("--min-validation-bets", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    frame = pd.read_csv(args.data, low_memory=False)
    result = analyze_segments(
        frame,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        min_validation_bets=args.min_validation_bets,
        top_n=args.top_n,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved market residual segment summary to {output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDE_DATA = PROJECT_ROOT / "data" / "datasets" / "wide_pair_data.csv"
DEFAULT_UMAREN_DATA = PROJECT_ROOT / "data" / "datasets" / "umaren_pair_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluations" / "experiments" / "pair_market_bias_sweep_summary.json"


@dataclass(frozen=True)
class PairConfig:
    name: str
    odds_col: str
    ninki_col: str
    hit_col: str
    gross_return_col: str


PAIR_CONFIGS = {
    "wide": PairConfig(
        name="wide",
        odds_col="WideOddsMeanDecimal",
        ninki_col="WideNinki",
        hit_col="WideHit",
        gross_return_col="WideGrossReturn",
    ),
    "umaren": PairConfig(
        name="umaren",
        odds_col="UmarenOddsDecimal",
        ninki_col="UmarenNinki",
        hit_col="UmarenHit",
        gross_return_col="UmarenGrossReturn",
    ),
}


DEFAULT_ODDS_BANDS = [
    (1.0, 3.0),
    (3.0, 5.0),
    (5.0, 8.0),
    (8.0, 12.0),
    (12.0, 20.0),
    (20.0, 50.0),
    (50.0, 999.0),
    (1.0, 999.0),
]
DEFAULT_NINKI_MAXES = [1, 2, 3, 5, 10, 20, 999]
DEFAULT_MODES = ["all", "favorite_in_band"]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def summarize_picks(picks: pd.DataFrame, config: PairConfig) -> dict[str, float | int]:
    bets = len(picks)
    stake = bets * 100.0
    gross_return = float(_coerce_numeric(picks.get(config.gross_return_col, pd.Series(dtype=float))).sum())
    return {
        "bets": int(bets),
        "hit_rate": float(_coerce_numeric(picks.get(config.hit_col, pd.Series(dtype=float))).mean()) if bets else 0.0,
        "return_rate": float(gross_return / stake * 100.0) if stake else 0.0,
    }


def load_pair_frame(path: Path, config: PairConfig) -> pd.DataFrame:
    columns = [
        "RaceDate",
        "RaceKey",
        "Umaban1",
        "Umaban2",
        config.odds_col,
        config.ninki_col,
        config.hit_col,
        config.gross_return_col,
    ]
    frame = pd.read_csv(path, usecols=columns, parse_dates=["RaceDate"], low_memory=False)
    frame = frame.dropna(subset=["RaceDate"]).copy()
    frame[config.odds_col] = _coerce_numeric(frame[config.odds_col], fill_value=-1.0)
    frame[config.ninki_col] = _coerce_numeric(frame[config.ninki_col], fill_value=999.0)
    frame[config.hit_col] = _coerce_numeric(frame[config.hit_col]).astype(int)
    frame[config.gross_return_col] = _coerce_numeric(frame[config.gross_return_col])
    return frame


def select_period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["RaceDate"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def select_policy_picks(
    frame: pd.DataFrame,
    config: PairConfig,
    odds_low: float,
    odds_high: float,
    ninki_max: int,
    mode: str,
) -> pd.DataFrame:
    selected = frame.loc[
        frame[config.odds_col].ge(odds_low)
        & frame[config.odds_col].lt(odds_high)
        & frame[config.ninki_col].le(ninki_max)
    ].copy()
    if selected.empty or mode == "all":
        return selected
    if mode != "favorite_in_band":
        raise ValueError(f"Unknown policy mode: {mode}")
    return (
        selected.sort_values(["RaceDate", "RaceKey", config.ninki_col, config.odds_col, "Umaban1", "Umaban2"])
        .groupby("RaceKey", as_index=False)
        .first()
    )


def summarize_yearly(picks: pd.DataFrame, config: PairConfig) -> list[dict[str, float | int]]:
    if picks.empty:
        return []
    rows = []
    for year, year_picks in picks.groupby(picks["RaceDate"].dt.year, sort=True):
        rows.append({"year": int(year), **summarize_picks(year_picks, config)})
    return rows


def sweep_pair_market_bias(
    frame: pd.DataFrame,
    config: PairConfig,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    min_validation_bets: int = 100,
    top_n: int = 20,
) -> list[dict[str, object]]:
    validation = select_period(frame, validation_start, validation_end)
    test = select_period(frame, test_start, test_end)
    rows: list[dict[str, object]] = []
    for odds_low, odds_high in DEFAULT_ODDS_BANDS:
        for ninki_max in DEFAULT_NINKI_MAXES:
            for mode in DEFAULT_MODES:
                validation_picks = select_policy_picks(validation, config, odds_low, odds_high, ninki_max, mode)
                if len(validation_picks) < min_validation_bets:
                    continue
                test_picks = select_policy_picks(test, config, odds_low, odds_high, ninki_max, mode)
                rows.append(
                    {
                        "odds_band": f"{odds_low:g}-{odds_high:g}",
                        "ninki_max": int(ninki_max),
                        "mode": mode,
                        "validation": summarize_picks(validation_picks, config),
                        "test": summarize_picks(test_picks, config),
                        "test_yearly": summarize_yearly(test_picks, config),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["validation"]["return_rate"],  # type: ignore[index]
            row["validation"]["bets"],  # type: ignore[index]
        ),
        reverse=True,
    )[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep simple final-odds pair-bet market-bias policies.")
    parser.add_argument("--wide-data", default=str(DEFAULT_WIDE_DATA))
    parser.add_argument("--umaren-data", default=str(DEFAULT_UMAREN_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-06-21")
    parser.add_argument("--min-validation-bets", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    result: dict[str, object] = {
        "validation_period": [args.validation_start, args.validation_end],
        "test_period": [args.test_start, args.test_end],
        "min_validation_bets": args.min_validation_bets,
        "top_n": args.top_n,
        "deployability": "research_final_odds_only",
    }
    for name, path in {"wide": Path(args.wide_data), "umaren": Path(args.umaren_data)}.items():
        config = PAIR_CONFIGS[name]
        frame = load_pair_frame(path, config)
        result[name] = sweep_pair_market_bias(
            frame,
            config,
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
    print(f"Saved pair market-bias sweep summary to {output_path}")


if __name__ == "__main__":
    main()

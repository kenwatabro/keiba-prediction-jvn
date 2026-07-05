import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "workstation_ml"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_pair_market_bias import (  # noqa: E402
    PAIR_CONFIGS,
    select_policy_picks,
    summarize_picks,
    sweep_pair_market_bias,
)


class PairMarketBiasTests(unittest.TestCase):
    def test_select_policy_picks_keeps_lowest_ninki_pair_per_race(self):
        config = PAIR_CONFIGS["wide"]
        frame = pd.DataFrame(
            [
                {
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "RaceKey": "r1",
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "WideOddsMeanDecimal": 4.0,
                    "WideNinki": 2,
                    "WideHit": 1,
                    "WideGrossReturn": 300,
                },
                {
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "RaceKey": "r1",
                    "Umaban1": 1,
                    "Umaban2": 3,
                    "WideOddsMeanDecimal": 3.5,
                    "WideNinki": 1,
                    "WideHit": 0,
                    "WideGrossReturn": 0,
                },
            ]
        )

        picks = select_policy_picks(frame, config, 3.0, 5.0, 2, "favorite_in_band")

        self.assertEqual(len(picks), 1)
        self.assertEqual(int(picks.iloc[0]["Umaban2"]), 3)

    def test_summarize_picks_calculates_return_rate(self):
        config = PAIR_CONFIGS["umaren"]
        picks = pd.DataFrame(
            [
                {"UmarenHit": 1, "UmarenGrossReturn": 500},
                {"UmarenHit": 0, "UmarenGrossReturn": 0},
            ]
        )

        summary = summarize_picks(picks, config)

        self.assertEqual(summary["bets"], 2)
        self.assertEqual(summary["hit_rate"], 0.5)
        self.assertEqual(summary["return_rate"], 250.0)

    def test_sweep_pair_market_bias_includes_test_yearly_breakdown(self):
        config = PAIR_CONFIGS["umaren"]
        rows = []
        for index in range(120):
            rows.append(
                {
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "RaceKey": f"v{index}",
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "UmarenOddsDecimal": 4.0,
                    "UmarenNinki": 1,
                    "UmarenHit": 1 if index == 0 else 0,
                    "UmarenGrossReturn": 500 if index == 0 else 0,
                }
            )
        for index in range(2):
            rows.append(
                {
                    "RaceDate": pd.Timestamp(f"202{5 + index}-01-01"),
                    "RaceKey": f"t{index}",
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "UmarenOddsDecimal": 4.0,
                    "UmarenNinki": 1,
                    "UmarenHit": 1,
                    "UmarenGrossReturn": 400,
                }
            )
        frame = pd.DataFrame(rows)

        result = sweep_pair_market_bias(
            frame,
            config,
            validation_start="2024-01-01",
            validation_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2026-12-31",
            min_validation_bets=100,
            top_n=1,
        )

        self.assertEqual(result[0]["odds_band"], "3-5")
        self.assertEqual(len(result[0]["test_yearly"]), 2)


if __name__ == "__main__":
    unittest.main()

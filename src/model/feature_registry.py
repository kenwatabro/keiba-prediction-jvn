from dataclasses import dataclass, field

FRIDAY_NO_MARKET_CONTRACT = "friday_no_market"
RACE_DAY_WEIGHT_CONTRACT = "race_day_weight"
LATE_MARKET_CONTRACT = "late_market"


@dataclass(frozen=True)
class FeatureRegistry:
    non_feature_columns: frozenset[str]
    market_feature_columns: frozenset[str]
    policy_only_columns: frozenset[str]
    raw_id_feature_columns: frozenset[str]
    categorical_columns: tuple[str, ...]
    availability_groups: dict[str, frozenset[str]] = field(default_factory=dict)
    availability_contract_exclusions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def availability_contract_names(self) -> tuple[str, ...]:
        return tuple(self.availability_contract_exclusions.keys())

    def excluded_availability_groups(self, availability_contract: str | None) -> tuple[str, ...]:
        if availability_contract is None:
            return ()
        if availability_contract not in self.availability_contract_exclusions:
            valid = ", ".join(self.availability_contract_names())
            raise ValueError(f"Unsupported availability contract: {availability_contract}. Valid values: {valid}")
        return self.availability_contract_exclusions[availability_contract]

    def excluded_availability_columns(self, availability_contract: str | None) -> frozenset[str]:
        excluded: set[str] = set()
        for group_name in self.excluded_availability_groups(availability_contract):
            if group_name not in self.availability_groups:
                raise ValueError(f"Availability group not found: {group_name}")
            excluded.update(self.availability_groups[group_name])
        return frozenset(excluded)

    def should_include_market_features(
        self,
        include_market_features: bool = False,
        availability_contract: str | None = None,
    ) -> bool:
        if availability_contract == LATE_MARKET_CONTRACT:
            return True
        if availability_contract is not None:
            self.excluded_availability_groups(availability_contract)
            return False
        return include_market_features

    def select_columns(
        self,
        columns: list[str],
        target_col: str,
        drop_raw_ids: bool = False,
        exclude_prefixes: list[str] | None = None,
        include_market_features: bool = False,
        availability_contract: str | None = None,
    ) -> list[str]:
        features = [col for col in columns if col not in self.non_feature_columns and col != target_col]
        availability_exclusions = self.excluded_availability_columns(availability_contract)
        features = [col for col in features if col not in availability_exclusions]
        if not self.should_include_market_features(
            include_market_features=include_market_features,
            availability_contract=availability_contract,
        ):
            features = [col for col in features if col not in self.market_feature_columns]
        features = [col for col in features if col not in self.policy_only_columns]
        if drop_raw_ids:
            features = [col for col in features if col not in self.raw_id_feature_columns]
        if exclude_prefixes:
            features = [col for col in features if not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        return features

    def explanation_metadata(self, feature_columns: list[str], default_top_k: int = 2) -> dict[str, object]:
        feature_set = set(feature_columns)
        return {
            "method": "lightgbm_pred_contrib",
            "score_space": "raw_margin",
            "default_top_k": default_top_k,
            "feature_display_names": {
                column: display_name
                for column, display_name in self.display_names.items()
                if column in feature_set
            },
            "feature_groups": {
                group_name: [column for column in columns if column in feature_set]
                for group_name, columns in self.groups.items()
                if any(column in feature_set for column in columns)
            },
        }


FEATURE_REGISTRY = FeatureRegistry(
    non_feature_columns=frozenset(
        {
            "RaceKey",
            "RaceDate",
            "Bamei",
            "KettoNum",
            "KakuteiJyuni",
            "Target",
            "TargetTop3",
            "TargetWin",
        }
    ),
    market_feature_columns=frozenset(
        {
            "OddsDecimal",
            "Ninki",
        }
    ),
    policy_only_columns=frozenset(
        {
            "JyokenName",
            "FutanBefore",
            "Blinker",
            "KisyuCodeBefore",
            "MinaraiCDBefore",
            "KyakusituKubun",
        }
    ),
    raw_id_feature_columns=frozenset(
        {
            "BanusiCode",
            "ChokyosiCode",
            "KisyuCode",
            "JCAfterKisyuCode",
            "JCBeforeKisyuCode",
        }
    ),
    availability_groups={
        "friday_available": frozenset(),
        "result_only": frozenset(
            {
                "NyusenTosu",
                "SyussoTosu",
            }
        ),
        "race_day_weight": frozenset(
            {
                "BaTaijyu",
                "ZogenSa",
                "WHAvailable",
                "WHHappyoTimeMinutes",
                "WHBaTaijyu",
                "WHZogenSa",
                "WHZogenSaAbs",
                "WHBaTaijyuDiffFromSE",
                "WHZogenSaDiffFromSE",
            }
        ),
        "race_day_weather": frozenset(
            {
                "TenkoCD",
                "SibaBabaCD",
                "DirtBabaCD",
                "TenkoBaba",
                "WEHenkoID",
                "WECurrentTenkoCD",
                "WECurrentSibaBabaCD",
                "WECurrentDirtBabaCD",
                "WEPreviousTenkoCD",
                "WEPreviousSibaBabaCD",
                "WEPreviousDirtBabaCD",
                "AVJiyuKubun",
                "CCAfterTrackCD",
                "CCBeforeTrackCD",
                "CCJiyuCd",
            }
        ),
        "late_market": frozenset(
            {
                "OddsDecimal",
                "Ninki",
            }
        ),
    },
    availability_contract_exclusions={
        FRIDAY_NO_MARKET_CONTRACT: (
            "result_only",
            "race_day_weight",
            "race_day_weather",
            "late_market",
        ),
        RACE_DAY_WEIGHT_CONTRACT: (
            "result_only",
            "race_day_weather",
            "late_market",
        ),
        LATE_MARKET_CONTRACT: (
            "result_only",
            "race_day_weather",
        ),
    },
    categorical_columns=(
        "JyoCD",
        "YoubiCD",
        "GradeCD",
        "JyuryoCD",
        "JyokenCD1",
        "JyokenCD2",
        "JyokenCD3",
        "JyokenCD4",
        "JyokenCD5",
        "DistanceBucket",
        "TrackCD",
        "CourseKubunCD",
        "TenkoCD",
        "SibaBabaCD",
        "DirtBabaCD",
        "TenkoBaba",
        "UmaKigoCD",
        "SexCD",
        "HinsyuCD",
        "KeiroCD",
        "TozaiCD",
        "ChokyosiCode",
        "BanusiCode",
        "KisyuCode",
        "MinaraiCD",
        "WEHenkoID",
        "WECurrentTenkoCD",
        "WECurrentSibaBabaCD",
        "WECurrentDirtBabaCD",
        "WEPreviousTenkoCD",
        "WEPreviousSibaBabaCD",
        "WEPreviousDirtBabaCD",
        "AVJiyuKubun",
        "JCAfterKisyuCode",
        "JCBeforeKisyuCode",
        "JCAfterMinaraiCD",
        "JCBeforeMinaraiCD",
        "CCAfterTrackCD",
        "CCBeforeTrackCD",
        "CCJiyuCd",
        "HCLastTresenKubun",
        "WCLastCourse",
        "WCLastBabaAround",
        "WCLastTresenKubun",
    ),
    display_names={
        "HorseWinRateBefore": "馬_勝率",
        "HorseTop3RateBefore": "馬_複勝率",
        "HorseAvgFinishPctBefore": "馬_平均着順率",
        "HorseLast3AvgFinishPct": "馬_近3走平均着順率",
        "HorseLast1FinishPct": "馬_前走着順率",
        "HorseLast3Top3Rate": "馬_近3走複勝率",
        "HorseLast3BestFinish": "馬_近3走最高着順",
        "HorseStartsBefore": "馬_出走数",
        "JockeyWinRateBefore": "騎手_勝率",
        "JockeyTop3RateBefore": "騎手_複勝率",
        "TrainerWinRateBefore": "調教師_勝率",
        "TrainerTop3RateBefore": "調教師_複勝率",
        "OwnerTop3RateSmoothBefore": "馬主_補正複勝率",
        "HorseRecentAvgFinish": "馬_近走平均着順",
        "HorseRecentTop3Rate": "馬_近走複勝率",
        "HorseRecentWinRate": "馬_近走勝率",
        "DistanceBucketTop3RateBefore": "距離帯_複勝率",
        "DistanceBucketWinRateBefore": "距離帯_勝率",
        "WeightDelta": "馬体重増減",
        "Futan": "斤量",
        "Barei": "馬齢",
        "Wakuban": "枠番",
        "Umaban": "馬番",
        "NyusenTosu": "入線頭数",
        "TorokuTosu": "登録頭数",
        "SyussoTosu": "出走頭数",
        "BaTaijyu": "馬体重",
        "ZogenSa": "馬体重増減",
        "JyoCD": "競馬場",
        "MinaraiCD": "見習区分",
        "TozaiCD": "東西",
        "Kyori": "距離",
        "OddsDecimal": "オッズ",
        "Ninki": "人気",
        "WHZogenSa": "当日馬体重増減",
    },
    groups={
        "horse_form": (
            "HorseWinRateBefore",
            "HorseTop3RateBefore",
            "HorseRecentAvgFinish",
            "HorseRecentTop3Rate",
            "HorseRecentWinRate",
        ),
        "jockey": (
            "JockeyWinRateBefore",
            "JockeyTop3RateBefore",
        ),
        "trainer": (
            "TrainerWinRateBefore",
            "TrainerTop3RateBefore",
        ),
        "race_context": (
            "JyoCD",
            "GradeCD",
            "Kyori",
            "DistanceBucket",
            "TrackCD",
            "CourseKubunCD",
            "TenkoBaba",
        ),
        "market": (
            "OddsDecimal",
            "Ninki",
        ),
        "race_day": (
            "WHZogenSa",
            "WECurrentTenkoCD",
            "WECurrentSibaBabaCD",
            "WECurrentDirtBabaCD",
        ),
    },
)

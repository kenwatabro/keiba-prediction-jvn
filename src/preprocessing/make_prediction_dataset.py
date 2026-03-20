import argparse

from make_dataset import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, make_prediction_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a prediction-ready dataset for pending JV-Link races.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw text files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for processed CSV output")
    parser.add_argument("--output-filename", default="prediction_data.csv", help="Output CSV filename")
    parser.add_argument("--prediction-date", default=None, help="Restrict output to a single race date (YYYY-MM-DD)")
    parser.add_argument("--include-o1", action="store_true", help="Use realtime O1 odds to fill pending-race market columns")
    parser.add_argument("--include-wh", action="store_true", help="Include WH body-weight bulletin features")
    parser.add_argument("--include-hc", action="store_true", help="Include HC hanro workout features")
    parser.add_argument("--include-wc", action="store_true", help="Include WC wood-chip workout features")
    args = parser.parse_args()

    make_prediction_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        prediction_date=args.prediction_date,
        include_o1=args.include_o1,
        include_wh=args.include_wh,
        include_hc=args.include_hc,
        include_wc=args.include_wc,
    )

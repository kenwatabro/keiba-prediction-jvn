import argparse
import logging
import sys
from pathlib import Path

if sys.platform != "win32":
    print("Error: This script must be run on Windows.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from data_loader.jvlink_client import JVLinkClient  # noqa: E402


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(description="Open JV-Link settings UI")
    parser.add_argument("--sid", default="PythonJVLink", help="JV-Link SID label")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional JV-Link local save path")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    client = JVLinkClient(sid=args.sid)

    try:
        client.initialize()
        if args.save_path:
            args.save_path.mkdir(parents=True, exist_ok=True)
            client.set_save_path(str(args.save_path))
        result = int(client.jv_link.JVSetUIProperties())
        logger.info("JVSetUIProperties returned: %s", result)
    finally:
        client.close()


if __name__ == "__main__":
    setup_logging()
    main()

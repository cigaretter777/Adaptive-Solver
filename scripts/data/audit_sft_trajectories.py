"""Audit structural validity and coverage of an SFT Parquet artifact."""

import argparse
from pathlib import Path

from adaptive_math.training.sft_io import audit_records, read_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min-records", type=int, default=0)
    args = parser.parse_args()
    audit = audit_records(read_records(args.input))
    print(audit.model_dump_json())
    if audit.total_records < args.min_records:
        parser.error(f"record count {audit.total_records} is below required minimum {args.min_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

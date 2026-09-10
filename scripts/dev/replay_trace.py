"""Verify and inspect a persisted AdaptiveMath-RL trajectory."""

import argparse
from pathlib import Path

import orjson

from adaptive_math.agent.replay import TraceEnvelope, replay, verify_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--verify-hash", action="store_true")
    parser.add_argument("--print-events", action="store_true")
    args = parser.parse_args()
    envelope = TraceEnvelope.model_validate_json(args.trace.read_bytes())
    if args.verify_hash and not verify_hash(envelope):
        parser.error("trace content hash mismatch")
    trajectory = replay(envelope, verify_content_hash=args.verify_hash)
    if args.print_events:
        for event in trajectory.events:
            print(orjson.dumps(event.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS).decode())
    else:
        print(orjson.dumps(trajectory.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

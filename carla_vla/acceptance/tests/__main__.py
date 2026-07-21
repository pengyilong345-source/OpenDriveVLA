"""Single command-line entry point for all D0 unit tests.

Usage:
  python -m carla_vla.acceptance.tests

Produces a text report (and writes output/carla_acceptance/D0_protocol/
unit_test_report.txt if that directory exists).
"""
from __future__ import annotations
import io
import json
import sys
import time
import unittest
from pathlib import Path


def main() -> int:
    loader = unittest.TestLoader()
    start_dir = Path(__file__).resolve().parent
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout,
                                      buffer=True)
    t0 = time.time()
    result = runner.run(suite)
    dt = time.time() - t0

    summary = {
        "tests_run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "ok": (len(result.errors) == 0 and len(result.failures) == 0),
        "duration_s": round(dt, 3),
    }
    print("\n" + "=" * 60)
    print(f"D0 unit tests  tests_run={summary['tests_run']} "
          f"errors={summary['errors']} failures={summary['failures']} "
          f"skipped={summary['skipped']} duration={summary['duration_s']}s "
          f"OK={summary['ok']}")
    print("=" * 60)

    # Write the report under output/carla_acceptance/D0_protocol/ if it
    # already exists; the orchestrator (D0.6) creates that dir.
    out_dir = Path("output/carla_acceptance/D0_protocol")
    if out_dir.exists():
        (out_dir / "unit_test_report.txt").write_text(
            f"D0 unit-test report\n"
            f"===================\n"
            f"tests_run = {summary['tests_run']}\n"
            f"errors    = {summary['errors']}\n"
            f"failures  = {summary['failures']}\n"
            f"skipped   = {summary['skipped']}\n"
            f"duration  = {summary['duration_s']} s\n"
            f"OK        = {summary['ok']}\n"
        )
        (out_dir / "unit_test_report.json").write_text(
            json.dumps(summary, indent=2) + "\n")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
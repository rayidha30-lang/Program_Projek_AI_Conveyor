"""Write a read-only quality report from manual rib dataset reviews."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_review_store import DatasetReviewStore


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    comparable = [row for row in rows if row["manual_label"] in {"OK", "NG"}]
    manual_ok = sum(row["manual_label"] == "OK" for row in comparable)
    manual_ng = sum(row["manual_label"] == "NG" for row in comparable)
    false_ng = sum(row["model_label"] == "NG" and row["manual_label"] == "OK" for row in comparable)
    false_ok = sum(row["model_label"] == "OK" and row["manual_label"] == "NG" for row in comparable)
    profiles: dict[str, int] = Counter(row["camera_profile"] for row in rows)
    return {
        "reviewed_total": len(rows),
        "invalid": sum(row["manual_label"] == "INVALID" for row in rows),
        "comparable_total": len(comparable),
        "model_ok_manual_ok": sum(row["model_label"] == "OK" and row["manual_label"] == "OK" for row in comparable),
        "model_ok_manual_ng": false_ok,
        "model_ng_manual_ok": false_ng,
        "model_ng_manual_ng": sum(row["model_label"] == "NG" and row["manual_label"] == "NG" for row in comparable),
        "false_ng": false_ng,
        "false_ok": false_ok,
        "false_ng_rate": false_ng / manual_ok if manual_ok else 0.0,
        "false_ok_rate": false_ok / manual_ng if manual_ng else 0.0,
        "invalid_rate": sum(row["manual_label"] == "INVALID" for row in rows) / len(rows) if rows else 0.0,
        "profiles": dict(profiles),
    }


def write_report(summary: dict[str, Any], output_dir: str | Path) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"rib_dataset_evaluation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.writer(report)
        writer.writerow(["metric", "value"])
        for key in (
            "reviewed_total", "comparable_total", "invalid", "model_ok_manual_ok",
            "model_ok_manual_ng", "model_ng_manual_ok", "model_ng_manual_ng",
            "false_ng", "false_ng_rate", "false_ok", "false_ok_rate", "invalid_rate",
        ):
            writer.writerow([key, summary[key]])
        for profile, count in sorted(summary["profiles"].items()):
            writer.writerow([f"camera_profile:{profile}", count])
    return str(report_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a read-only rib dataset quality CSV.")
    parser.add_argument("--db", required=True, help="Path to AI_Slat_Rib_Conveyor.db")
    parser.add_argument("--output-dir", default="Database/reports", help="Directory for the CSV report")
    args = parser.parse_args()
    summary = summarize_rows(DatasetReviewStore(args.db).evaluation_rows())
    report_path = write_report(summary, args.output_dir)
    print(f"Reviewed: {summary['reviewed_total']} | Comparable: {summary['comparable_total']} | INVALID: {summary['invalid']}")
    print(f"False NG: {summary['false_ng_rate']:.2%} | False OK: {summary['false_ok_rate']:.2%} | Invalid: {summary['invalid_rate']:.2%}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

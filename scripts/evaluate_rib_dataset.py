"""Write a read-only quality report from manual rib dataset reviews."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conveyor_db_path import resolve_db_path
from dataset_review_store import DatasetReviewStore

THRESHOLD_MAPPINGS = (
    ("pv_flatness", "rib_pv_threshold"),
    ("sagitta", "rib_sagitta_threshold"),
    ("inner_dev", "rib_dev_threshold"),
    ("tip_dev", "rib_tip_threshold"),
)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _distribution_stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 2),
        "median": round(_percentile(ordered, 50.0), 2),
        "p95": round(_percentile(ordered, 95.0), 2),
        "max": round(ordered[-1], 2),
    }


def _compute_threshold_analytics(
    comparable: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, dict[str, Any]]]:
    distributions: dict[str, dict[str, dict[str, float | int]]] = {"OK": {}, "NG": {}}
    recommendations: dict[str, dict[str, Any]] = {}

    for metric_key, config_key in THRESHOLD_MAPPINGS:
        ok_vals = [
            float(row[metric_key])
            for row in comparable
            if row["manual_label"] == "OK" and row.get(metric_key) is not None
        ]
        ng_vals = [
            float(row[metric_key])
            for row in comparable
            if row["manual_label"] == "NG" and row.get(metric_key) is not None
        ]
        if ok_vals:
            distributions["OK"][metric_key] = _distribution_stats(ok_vals)
        if ng_vals:
            distributions["NG"][metric_key] = _distribution_stats(ng_vals)
        if not ok_vals:
            continue
        ok_p95 = float(distributions["OK"][metric_key]["p95"])
        if ng_vals:
            ng_min = float(distributions["NG"][metric_key]["min"])
            if ok_p95 < ng_min:
                rec_val = round((ok_p95 + ng_min) / 2.0, 2)
                status = "clean_separation"
            else:
                rec_val = round(ok_p95 * 1.05, 2)
                status = "overlap_ok_ng"
            recommendations[config_key] = {
                "value": rec_val,
                "status": status,
                "ok_p95": ok_p95,
                "ng_min": ng_min,
            }
        else:
            recommendations[config_key] = {
                "value": round(ok_p95 * 1.15, 2),
                "status": "ok_only_margin",
                "ok_p95": ok_p95,
                "ng_min": None,
            }
    return distributions, recommendations


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    comparable = [row for row in rows if row["manual_label"] in {"OK", "NG"}]
    manual_ok = sum(row["manual_label"] == "OK" for row in comparable)
    manual_ng = sum(row["manual_label"] == "NG" for row in comparable)
    false_ng = sum(row["model_label"] == "NG" and row["manual_label"] == "OK" for row in comparable)
    false_ok = sum(row["model_label"] == "OK" and row["manual_label"] == "NG" for row in comparable)
    profiles: dict[str, int] = Counter(row["camera_profile"] for row in rows)
    distributions, recommendations = _compute_threshold_analytics(comparable)
    false_cases = [
        {
            "dataset_id": row.get("dataset_id"),
            "type": "FALSE_NG" if row["model_label"] == "NG" else "FALSE_OK",
            "pv_flatness": row.get("pv_flatness"),
            "sagitta": row.get("sagitta"),
            "inner_dev": row.get("inner_dev"),
            "tip_dev": row.get("tip_dev"),
        }
        for row in comparable
        if row["model_label"] != row["manual_label"]
    ]
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
        "metric_distributions": distributions,
        "threshold_recommendations": recommendations,
        "false_cases": false_cases,
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
        for group_label, metrics in summary.get("metric_distributions", {}).items():
            for metric_key, stats in sorted(metrics.items()):
                for stat_name, stat_val in stats.items():
                    writer.writerow([f"metric_dist:{group_label}:{metric_key}:{stat_name}", stat_val])
        for config_key, rec in sorted(summary.get("threshold_recommendations", {}).items()):
            writer.writerow([f"recommended:{config_key}", rec["value"]])
            writer.writerow([f"recommended_status:{config_key}", rec["status"]])
    return str(report_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a read-only rib dataset quality CSV.")
    parser.add_argument("--db", default=resolve_db_path(), help="Path to AI_Slat_Rib_Conveyor.db")
    parser.add_argument("--output-dir", default="Database/reports", help="Directory for the CSV report")
    args = parser.parse_args()
    summary = summarize_rows(DatasetReviewStore(args.db).evaluation_rows())
    report_path = write_report(summary, args.output_dir)
    print(f"Reviewed: {summary['reviewed_total']} | Comparable: {summary['comparable_total']} | INVALID: {summary['invalid']}")
    print(f"False NG: {summary['false_ng_rate']:.2%} | False OK: {summary['false_ok_rate']:.2%} | Invalid: {summary['invalid_rate']:.2%}")
    recs = summary.get("threshold_recommendations", {})
    if recs:
        print("Threshold Recommendations (read-only):")
        for cfg_key, rec in sorted(recs.items()):
            print(f"  - {cfg_key}: {rec['value']} ({rec['status']}, OK_p95={rec['ok_p95']}, NG_min={rec['ng_min']})")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train Jalur 1 pure-NumPy tip classifier from reviewed Rib_Dataset samples."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_SRC_DIR = PROJECT_ROOT / "Vision_Engine_Core" / "02_Top_Inspection_Slat_Rib" / "src"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "Vision_Engine_Core" / "02_Top_Inspection_Slat_Rib" / "configs" / "rib_tip_classifier.json"
)

for p in (PROJECT_ROOT, TOP_SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from conveyor_db_path import resolve_db_path
from rib_tip_features import FEATURE_NAMES, extract_tip_classifier_features


def load_reviewed_dataset(
    db_path: str | Path,
    camera_profile: str | None = None,
    return_groups: bool = False,
):
    """Fetch manually reviewed OK/NG samples with ProfileJson from SQLite."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        query = """
            SELECT d.DatasetID, r.ManualLabel, d.CameraProfile,
                   d.PVFlatness, d.Sagitta, d.InnerDev, d.TipDev, d.ProfileJson, d.RibNumber
            FROM Rib_Dataset AS d
            INNER JOIN Rib_Dataset_Review AS r ON r.DatasetID = d.DatasetID
            WHERE r.ManualLabel IN ('OK', 'NG') AND d.ProfileJson IS NOT NULL
        """
        params: list[str] = []
        if camera_profile:
            query += " AND d.CameraProfile = ?"
            params.append(camera_profile)
        query += " ORDER BY d.DatasetID ASC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    features_list = []
    labels_list = []
    groups_list = []
    group_map: dict[tuple, int] = {}
    for _did, man_label, _prof, pv, sag, idev, tdev, pjson, rib_no in rows:
        try:
            pts = json.loads(pjson)
        except Exception:
            continue
        feat = extract_tip_classifier_features(
            pts,
            float(pv or 0.0),
            float(sag or 0.0),
            float(idev or 0.0),
            float(tdev or 0.0),
        )
        if feat is not None:
            features_list.append(feat)
            labels_list.append(1 if str(man_label).upper() == "NG" else 0)
            g_key = (
                int(rib_no) if rib_no is not None else -1,
                round(float(pv or 0.0), 2),
                round(float(sag or 0.0), 2),
                round(float(idev or 0.0), 2),
            )
            if g_key not in group_map:
                group_map[g_key] = len(group_map)
            groups_list.append(group_map[g_key])

    if not features_list:
        empty_X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int32)
        if return_groups:
            return empty_X, empty_y, np.empty((0,), dtype=np.int32)
        return empty_X, empty_y
    X_arr = np.stack(features_list).astype(np.float32)
    y_arr = np.array(labels_list, dtype=np.int32)
    if return_groups:
        return X_arr, y_arr, np.array(groups_list, dtype=np.int32)
    return X_arr, y_arr


def train_and_export(
    db_path: str | Path,
    out_path: str | Path = DEFAULT_OUT_PATH,
    camera_profile: str | None = None,
    prob_threshold: float = 0.50,
) -> dict:
    """Run StratifiedGroupKFold CV evaluation, train final LogisticRegression, and export JSON."""
    X, y, groups = load_reviewed_dataset(db_path, camera_profile=camera_profile, return_groups=True)
    n_ok = int((y == 0).sum())
    n_ng = int((y == 1).sum())
    if n_ok < 5 or n_ng < 2:
        raise ValueError(
            f"Sampel review belum cukup untuk training (OK={n_ok}, NG={n_ng}; butuh minimal 5 OK & 2 NG)."
        )

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )

    n_ok_groups = int(len(set(groups[y == 0])))
    n_ng_groups = int(len(set(groups[y == 1])))
    n_splits = min(5, n_ok_groups, n_ng_groups)
    cv_metrics: dict[str, float | int] = {}
    if n_splits >= 2:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        probs = cross_val_predict(pipe, X, y, cv=sgkf, groups=groups, method="predict_proba")[:, 1]
        preds = (probs >= prob_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        auc = roc_auc_score(y, probs)
        cv_metrics = {
            "cv_folds": int(n_splits),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc), 4),
        }

    pipe.fit(X, y)
    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    clf: LogisticRegression = pipe.named_steps["logisticregression"]

    payload = {
        "model_type": "logistic_regression_standard_scaled",
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "weights": [round(float(v), 6) for v in clf.coef_[0]],
        "bias": round(float(clf.intercept_[0]), 6),
        "prob_threshold": float(prob_threshold),
        "metadata": {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "db_path": str(Path(db_path).resolve()),
            "camera_profile": camera_profile or "ALL",
            "n_total": int(len(y)),
            "n_ok": n_ok,
            "n_ng": n_ng,
            **cv_metrics,
        },
    }

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Jalur 1 tip classifier from reviewed Rib_Dataset.")
    parser.add_argument("--db", default=resolve_db_path(), help="Path to AI_Slat_Rib_Conveyor.db")
    parser.add_argument("--profile", default=None, help="Optional CameraProfile filter")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output JSON model path")
    parser.add_argument("--prob-threshold", type=float, default=0.50, help="NG probability cutoff")
    args = parser.parse_args()

    payload = train_and_export(
        args.db,
        out_path=args.out,
        camera_profile=args.profile,
        prob_threshold=args.prob_threshold,
    )
    meta = payload["metadata"]
    print(f"Trained Jalur 1 Classifier: OK={meta['n_ok']} | NG={meta['n_ng']} -> {args.out}")
    if "f1" in meta:
        print(
            f"CV ({meta['cv_folds']}-Fold): TP={meta['tp']} TN={meta['tn']} "
            f"FP={meta['fp']} FN={meta['fn']} | Prec={meta['precision']:.3f} "
            f"Rec={meta['recall']:.3f} F1={meta['f1']:.3f} AUC={meta['auc']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

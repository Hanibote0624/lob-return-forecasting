#!/usr/bin/env python3
"""Run real Stage1–5 commands and verify every small synthetic label/window."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.synthetic_data import (
    FACTORS,
    SPLITS,
    artifact_path,
    create_fixture,
    read_json,
    read_jsonl,
    reference_session,
    write_json,
)
from src.configuration import load_config, validate_config
from src.window_loader import LoaderConfig, iter_batches_with_time


PREPARATION_STAGES = ("manifest", "labels", "pack", "windows", "healthcheck")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def identity(item):
    return f"{item['stock_code']}_{item['date']}_{item['session']}"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_pipeline(config_path, stages=PREPARATION_STAGES):
    """Use the public runner, including configuration and dependency preflight."""
    config_path = Path(config_path).resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_pipeline.py"),
            "--config-path",
            str(config_path),
            "--stages",
            *stages,
        ],
        cwd=config_path.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (config_path.parent / "pipeline.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    return result


def verify_outputs(cfg, sessions):
    """Compare produced artifacts with references computed from source CSVs."""
    refs = {identity(item): reference_session(item["path"], cfg) for item in sessions}
    expected_ids = set(refs)
    manifest = read_jsonl(artifact_path(cfg, "stage1", "manifest_ok_path"))
    require(len(manifest) == len(expected_ids), "Stage1 session count mismatch")
    require({identity(item) for item in manifest} == expected_ids, "Stage1 identity mismatch")
    require(not read_jsonl(artifact_path(cfg, "stage1", "manifest_bad_path")), "bad Stage1 inputs")
    require(
        all(item["n_rows"] == len(refs[identity(item)]["X"]) for item in manifest),
        "Stage1 row count mismatch",
    )
    stage1_summary = read_json(artifact_path(cfg, "stage1", "summary_path"))
    for split in SPLITS:
        require(
            stage1_summary["splits"][f"{split}_count"]
            == sum(s["split"] == split for s in sessions),
            f"Stage1 split mismatch: {split}",
        )
    schema = read_json(artifact_path(cfg, "stage1", "schema_path"))
    require(schema["factor_cols"] == FACTORS, "Stage1 factor order mismatch")

    train_refs = [refs[identity(item)] for item in sessions if item["split"] == "train"]
    train_returns = np.concatenate(
        [ref["r_raw"][ref["is_valid"].astype(bool)] for ref in train_refs]
    )
    require(
        len(train_returns) >= 1000, "fixture must preserve the production quantile sample minimum"
    )
    q90, q99, q100 = np.quantile(np.abs(train_returns).astype(np.float64), [0.90, 0.99, 1.0])
    stats = read_json(artifact_path(cfg, "stage2", "label_stats_path"))
    for key, expected in zip(
        ("q90_abs_r_raw", "q99_abs_r_raw", "q100_abs_r_raw"), (q90, q99, q100)
    ):
        np.testing.assert_allclose(
            stats["train_only_quantiles"][key], expected, rtol=2e-6, atol=2e-10
        )
    scale = cfg["label"]["fixed_scale"]
    limit = min(q99 * scale, cfg["stage2"]["max_clip_abs_scaled"])
    np.testing.assert_allclose(stats["clip_abs_scaled"], limit, rtol=2e-6, atol=2e-10)
    require(stats["scale"] == scale, "configured scale was overridden")
    index = read_jsonl(artifact_path(cfg, "stage2", "labels_index_path"))
    require(
        len(index) == len(sessions) and {identity(item) for item in index} == expected_ids,
        "Stage2 session identities mismatch",
    )
    wanted_splits = {identity(item): item["split"] for item in sessions}
    clipped = dict.fromkeys(SPLITS, 0)
    numeric_hash = hashlib.sha256()

    def hash_array(name, array):
        array = np.asarray(array)
        numeric_hash.update(f"{name}:{array.dtype}:{array.shape}".encode())
        numeric_hash.update(array.tobytes())

    for item in index:
        sid = identity(item)
        ref = refs[sid]
        require(item["split"] == wanted_splits[sid], f"Stage2 wrong split: {sid}")
        valid = ref["is_valid"].astype(bool)
        scaled_unclipped = (ref["r_raw"].astype(np.float64) * scale).astype(np.float32)
        ref["r_scaled"] = np.clip(scaled_unclipped, -limit, limit).astype(np.float32)
        ref["was_clipped"] = (valid & (np.abs(scaled_unclipped) > limit)).astype(np.uint8)
        sw = cfg["sample_weight"]
        ref["w"] = np.array(
            [
                min(
                    sw["max_weight"],
                    max(
                        sw["min_weight"],
                        1 + sw["alpha"] * min(abs(float(value)) / q90, sw["clip_max"]),
                    ),
                )
                if ok
                else 0
                for value, ok in zip(ref["r_raw"], valid)
            ],
            dtype=np.float32,
        )
        require(
            item["n"] == len(valid) and item["valid"] == int(valid.sum()), f"Stage2 counts: {sid}"
        )
        for field in ("raw_label_path", "final_label_path"):
            with np.load(item[field], allow_pickle=False) as labels:
                require(int(labels["label_contract_version"]) == 2, "wrong label contract version")
                require(labels["t_sec"].dtype == np.float64, "timestamps lost float64 precision")
                require(labels["mid"].dtype == np.float64, "midpoint precision was truncated")
                for key in (
                    "r_raw",
                    "t_sec",
                    "is_valid",
                    "invalid_reason",
                    "future_count",
                    "lo",
                    "hi",
                ):
                    np.testing.assert_allclose(
                        labels[key],
                        ref[key],
                        rtol=2e-6,
                        atol=2e-10,
                        equal_nan=True,
                        err_msg=f"{sid}/{field}/{key}",
                    )
                np.testing.assert_allclose(
                    labels["mid"],
                    ref["mid"],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{sid}: unrounded midpoint",
                )
                require(np.isnan(labels["r_raw"][~valid]).all(), "invalid labels must remain NaN")
                if field == "final_label_path":
                    for key in ("r_scaled", "was_clipped"):
                        np.testing.assert_allclose(
                            labels[key],
                            ref[key],
                            rtol=2e-6,
                            atol=2e-10,
                            equal_nan=True,
                            err_msg=f"{sid}/{key}",
                        )
                    for key in sorted(labels.files):
                        hash_array(f"{sid}/{key}", labels[key])
        clipped[item["split"]] += int(ref["was_clipped"].sum())
    require(clipped["val"] > 0 and clipped["test"] > 0, "held-out fixture must exercise clipping")

    stage2_summary = read_json(artifact_path(cfg, "stage2", "stage2_summary_path"))
    invalid_counts = {
        name: sum(int(np.count_nonzero(ref["invalid_reason"] & flag)) for ref in refs.values())
        for name, flag in (
            ("invalid_current_mid", 1),
            ("incomplete_horizon", 2),
            ("insufficient_future", 4),
            ("nonfinite_return", 8),
        )
    }
    require(stage2_summary["counts"]["invalid_reasons"] == invalid_counts, "invalid reason totals")
    require(
        all(invalid_counts[key] > 0 for key in invalid_counts if key != "nonfinite_return"),
        "fixture must exercise missing prices, incomplete horizons and empty futures",
    )

    stage3 = read_json(artifact_path(cfg, "stage3", "packs_manifest_path"))
    stage4_path = artifact_path(cfg, "stage4", "stage4_manifest_path")
    stage4 = read_json(stage4_path)
    final_schema_path = artifact_path(cfg, "stage3", "final_schema_path")
    final_schema = read_json(final_schema_path)
    require(
        final_schema["factor_cols"] == FACTORS and stage4["num_factors"] == len(FACTORS),
        "factor schema changed during packing",
    )
    require(
        stage3["norm_map_path"] is None and not final_schema["dropped_factors"],
        "single-stock fixture unexpectedly normalized or removed factors",
    )
    input_stats = read_json(artifact_path(cfg, "stage3", "input_stats_path"))
    expected_train_X = np.concatenate(
        [ref["X"][ref["is_valid"].astype(bool)] for ref in train_refs]
    ).astype(np.float64)
    expected_mean = expected_train_X.mean(axis=0)
    expected_std = np.sqrt(np.maximum(expected_train_X.var(axis=0, ddof=1), 1e-12))
    np.testing.assert_allclose(input_stats["mean"], expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(input_stats["std"], expected_std, rtol=1e-12, atol=1e-12)
    require(
        input_stats["train_row_count"] == len(expected_train_X), "wrong train statistics row count"
    )
    require(
        input_stats["factor_schema_sha256"]
        == hashlib.sha256(final_schema_path.read_bytes()).hexdigest(),
        "statistics do not bind the generated factor schema",
    )
    split_summary = {}
    window = cfg["features"]["window_W"]
    for split in SPLITS:
        chosen = [item for item in sessions if item["split"] == split]
        expected_ends = [
            (identity(item), end) for item in chosen for end in refs[identity(item)]["endpoints"]
        ]
        expected_rows = sum(len(refs[identity(item)]["X"]) for item in chosen)
        require(stage3["counts"]["rows"][split] == expected_rows, f"packed row count: {split}")
        require(
            stage3["counts"]["sessions"][split] == len(chosen), f"packed session count: {split}"
        )
        require(
            stage3["counts"]["valid_end_rows_for_windowing"][split] == len(expected_ends),
            f"Stage3 window count: {split}",
        )
        segment_ids = []
        for shard in stage3["shards"][split]:
            segments = read_jsonl(shard["files"]["segments"])
            require(
                sum(seg["length"] for seg in segments) == shard["valid_rows"], "used shard rows"
            )
            require(
                shard["valid_rows"] < shard["capacity_rows"],
                "fixture must exercise unused shard tails",
            )
            cursor = 0
            arrays = {
                key: np.load(path, allow_pickle=False)
                for key, path in shard["files"].items()
                if key != "segments"
            }
            for seg in segments:
                sid = identity(seg)
                segment_ids.append(sid)
                ref = refs[sid]
                require(
                    seg["start_row"] == cursor and seg["length"] == len(ref["X"]), "segment bounds"
                )
                require(seg["valid_end_count"] == len(ref["endpoints"]), "segment window count")
                stop = cursor + seg["length"]
                for key, expected_key in (
                    ("X", "X"),
                    ("y", "r_scaled"),
                    ("y_raw", "r_raw"),
                    ("w", "w"),
                    ("is_valid", "is_valid"),
                    ("t_sec", "t_sec"),
                ):
                    np.testing.assert_allclose(
                        arrays[key][cursor:stop],
                        ref[expected_key],
                        rtol=2e-6,
                        atol=2e-10,
                        equal_nan=True,
                        err_msg=f"pack/{sid}/{key}",
                    )
                    hash_array(f"pack/{sid}/{key}", arrays[key][cursor:stop])
                cursor = stop
        require(
            Counter(segment_ids) == Counter(identity(item) for item in chosen),
            "lost/duplicated sessions",
        )
        produced_ends = []
        for block in stage4["splits"][split]["blocks"]:
            sid = block["seg_session_id"]
            require(block["split"] == split and sid in refs, "block identity")
            require(block["end_len"] <= cfg["stage4"]["block_size_ends"], "oversized block")
            start = block["end_start"] - block["seg_start_row"]
            produced_ends.extend((sid, end) for end in range(start, start + block["end_len"]))
        require(
            produced_ends == expected_ends, f"missing/duplicated/cross-session windows: {split}"
        )
        require(
            stage4["counts"][split]["total_end_positions"] == len(expected_ends), "Stage4 count"
        )
        batches = list(
            iter_batches_with_time(
                LoaderConfig(
                    project_root=cfg["project"]["project_root"],
                    stage4_manifest_path=str(stage4_path),
                    split=split,
                    batch_size=64,
                    window_W=window,
                    num_factors=len(FACTORS),
                    shuffle_blocks=False,
                    drop_remainder=False,
                )
            )
        )
        require(len(expected_ends) % 64 != 0, "fixture must exercise partial evaluation batches")
        wanted = [[], [], [], []]
        for sid, end in expected_ends:
            ref = refs[sid]
            begin = end - window + 1
            times = ref["t_sec"][begin : end + 1]
            for bucket, value in zip(
                wanted,
                (
                    ref["X"][begin : end + 1],
                    (times - times[0]).astype(np.float32),
                    ref["r_scaled"][end],
                    ref["w"][end],
                ),
            ):
                bucket.append(value)
        for column in range(4):
            actual = np.concatenate([batch[column] for batch in batches])
            np.testing.assert_allclose(actual, np.asarray(wanted[column]), rtol=2e-6, atol=2e-10)
        split_summary[split] = {
            "sessions": len(chosen),
            "rows": expected_rows,
            "valid_labels": sum(int(refs[identity(item)]["is_valid"].sum()) for item in chosen),
            "windows": len(expected_ends),
            "shards": len(stage3["shards"][split]),
            "evaluation_batches": len(batches),
            "last_batch_size": len(batches[-1][2]),
            "clipped_labels": clipped[split],
            "span_exclusions": sum(refs[identity(item)]["span_exclusions"] for item in chosen),
            "gap_exclusions": sum(refs[identity(item)]["gap_exclusions"] for item in chosen),
        }
        require(
            split_summary[split]["span_exclusions"] > 0
            and split_summary[split]["gap_exclusions"] > 0,
            "fixture must exercise both history-time guards",
        )
    require(split_summary["train"]["shards"] > 1, "fixture must exercise shard rollover")
    health = read_json(artifact_path(cfg, "stage5", "report_path"))
    require(health["ok"] and health["status"] == "passed", "healthcheck did not pass")
    require(
        health["strict"] and all(health["consistency"].values()), "healthcheck metadata mismatch"
    )
    for split in SPLITS:
        part = health["splits"][split]
        require(part["ok"] and not any(part["violations"].values()), f"healthcheck failed: {split}")
        require(not any(part["window_checks"].values()), f"healthcheck windows failed: {split}")
    raw_hash = hashlib.sha256()
    for item in sessions:
        raw_hash.update(identity(item).encode())
        raw_hash.update(Path(item["path"]).read_bytes())
    return {
        "splits": split_summary,
        "invalid_reasons": invalid_counts,
        "train_quantiles": stats["train_only_quantiles"],
        "training_fingerprint": digest_json(
            {
                "quantiles": stats["train_only_quantiles"],
                "mean": input_stats["mean"],
                "std": input_stats["std"],
                "rows": input_stats["train_row_count"],
            }
        ),
        "source_csv_sha256": raw_hash.hexdigest(),
        "numeric_artifact_sha256": numeric_hash.hexdigest(),
    }


def run_scenarios(output_root):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for name, variant in (("baseline", False), ("held_out_changed", True)):
        cfg, config_path, sessions = create_fixture(output_root / name, held_out_variant=variant)
        validate_config(load_config(config_path))
        result = run_pipeline(config_path)
        require(
            result.returncode == 0, f"{name} pipeline failed:\n{result.stdout}\n{result.stderr}"
        )
        cases[name] = verify_outputs(cfg, sessions)
    baseline, changed = cases["baseline"], cases["held_out_changed"]
    require(
        baseline["training_fingerprint"] == changed["training_fingerprint"],
        "held-out data leaked into training statistics",
    )
    require(
        baseline["numeric_artifact_sha256"] != changed["numeric_artifact_sha256"],
        "held-out perturbation had no effect",
    )
    source_paths = [
        "src/target_contract.py",
        "src/stage1_make_manifest_multi.py",
        "src/stage2_build_labels_clear.py",
        "src/stage3_pack_rows.py",
        "src/stage4_build_stage4_manifest.py",
        "src/stage5_dataset_healthcheck.py",
        "src/data_contract.py",
        "src/feature_preprocessing.py",
        "src/window_loader.py",
        "src/configuration.py",
        "src/pipeline.py",
        "scripts/synthetic_data.py",
        "scripts/check_synthetic_pipeline.py",
    ]
    summary = {
        "status": "passed",
        "scope": "synthetic Stage1-5 and window-loader verification",
        "private_data_used": False,
        "tensorflow_used": False,
        "gpu_training_verified": False,
        "github_hosted_ci_verified": False,
        "stages": list(PREPARATION_STAGES),
        "fixture": {
            "stock_code": "000000",
            "factors": FACTORS,
            "window_size": 8,
            "label_interval_seconds": [2.5, 3.5],
            "label_scale": 37.0,
        },
        "environment": {
            "python": sys.version.split()[0],
            **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
        },
        "checks": {
            "all_labels_compared_to_reference": True,
            "all_windows_compared_to_reference": True,
            "train_statistics_unchanged_after_held_out_perturbation": True,
        },
        "cases": cases,
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths
        },
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Retain generated cases/logs/report in a NEW directory."
    )
    args = parser.parse_args(argv)
    try:
        if args.output_dir is not None:
            summary = run_scenarios(args.output_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="lob-synthetic-") as temp:
                summary = run_scenarios(Path(temp) / "run")
        print(json.dumps(summary, indent=2, allow_nan=False))
    except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Synthetic verification failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

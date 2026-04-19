from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from src.asr.model import ConvBiGRUCTC
from src.asr.tokenizer import build_tokenizer
from src.train_baseline import count_parameters


@dataclass(frozen=True)
class MetricConfig:
    key: str
    fallback_key: str | None
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable hyperparameter sweep (grid/random/optuna) "
            "from YAML config and save checkpoint-level dev metrics table."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sweep_baseline.yaml"),
        help="Path to sweep YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned runs from config, do not launch training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    if not isinstance(loaded, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    return loaded


def ensure_mapping(raw: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Config key '{key}' must be a mapping/object.")
    return raw


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got: {value!r}")


def cast_value(value: Any, value_type: str) -> Any:
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "bool":
        return as_bool(value)
    if value_type in {"str", "categorical"}:
        return value
    raise ValueError(f"Unsupported parameter type: {value_type!r}")


def estimate_num_params(
    *,
    tokenizer_type: str,
    n_mels: int,
    encoder_dim: int,
    encoder_layers: int,
    dropout: float,
) -> int:
    tokenizer = build_tokenizer({"type": tokenizer_type, "blank_id": 0})
    model = ConvBiGRUCTC(
        n_mels=n_mels,
        vocab_size=tokenizer.vocab_size,
        encoder_dim=encoder_dim,
        encoder_layers=encoder_layers,
        dropout=dropout,
    )
    return count_parameters(model)


def estimate_num_params_from_train_args(train_args: dict[str, Any]) -> int | None:
    try:
        tokenizer_type = str(train_args.get("tokenizer", "russian_number_words"))
        n_mels = int(train_args.get("n_mels", 80))
        encoder_dim = int(train_args.get("encoder_dim", 192))
        encoder_layers = int(train_args.get("encoder_layers", 2))
        dropout = float(train_args.get("dropout", 0.1))
    except (TypeError, ValueError):
        return None
    try:
        return estimate_num_params(
            tokenizer_type=tokenizer_type,
            n_mels=n_mels,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            dropout=dropout,
        )
    except Exception:
        return None


def normalize_method(raw_method: Any) -> str:
    method = str(raw_method or "grid_search").strip().lower()
    aliases = {
        "grid": "grid_search",
        "grid_search": "grid_search",
        "gird_search": "grid_search",
        "random": "random_search",
        "random_search": "random_search",
        "optuna": "optuna",
    }
    normalized = aliases.get(method)
    if normalized is None:
        supported = sorted(set(aliases.values()))
        raise ValueError(f"Unsupported sweep method: {method!r}. Supported: {supported}")
    return normalized


def metric_value(row: dict[str, Any], metric_cfg: MetricConfig) -> float | None:
    value = row.get(metric_cfg.key)
    if value is None and metric_cfg.fallback_key:
        value = row.get(metric_cfg.fallback_key)
    if value is None:
        return None
    return float(value)


def metric_sort_key(value: float | None, mode: str) -> float:
    if value is None:
        return float("inf")
    if mode == "min":
        return value
    return -value


def is_better(candidate: float | None, best: float | None, mode: str) -> bool:
    if candidate is None:
        return False
    if best is None:
        return True
    if mode == "min":
        return candidate < best
    return candidate > best


def sanitize_run_part(value: Any) -> str:
    text = str(value)
    sanitized = re.sub(r"[^0-9a-zA-Z_-]+", "_", text).strip("_")
    return sanitized or "value"


def run_label(trial_params: dict[str, Any]) -> str:
    if not trial_params:
        return "default"
    parts = [
        f"{sanitize_run_part(key)}-{sanitize_run_part(value)}"
        for key, value in sorted(trial_params.items())
    ]
    return "__".join(parts)[:140]


def to_cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_train_command(train_module: str, train_args: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", train_module]
    for key, value in train_args.items():
        if value is None:
            continue
        flag = to_cli_flag(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            serialized = ",".join(str(item) for item in value)
            cmd.extend([flag, serialized])
            continue
        cmd.extend([flag, str(value)])
    return cmd


def precheck_max_parameters(train_args: dict[str, Any]) -> tuple[bool, int | None]:
    max_params_raw = train_args.get("max_parameters")
    if max_params_raw is None:
        return False, None
    try:
        max_params = int(max_params_raw)
    except (TypeError, ValueError):
        return False, None
    estimated = estimate_num_params_from_train_args(train_args)
    if estimated is None:
        return False, None
    return estimated > max_params, estimated


def normalize_space_specs(raw_space: Any) -> dict[str, dict[str, Any]]:
    space_cfg = ensure_mapping(raw_space, key="sweep.params")
    normalized: dict[str, dict[str, Any]] = {}
    for param_name, raw_spec in space_cfg.items():
        spec = ensure_mapping(raw_spec, key=f"sweep.params.{param_name}")
        value_type = str(spec.get("type", "categorical")).strip().lower()
        if value_type not in {"int", "float", "bool", "str", "categorical"}:
            raise ValueError(
                f"Unsupported type for param {param_name!r}: {value_type!r}"
            )
        normalized[param_name] = {"type": value_type, **spec}
    if not normalized:
        raise ValueError("sweep.params must contain at least one parameter.")
    return normalized


def spec_values_for_grid(param_name: str, spec: dict[str, Any]) -> list[Any]:
    value_type = str(spec["type"])
    if "values" in spec:
        raw_values = spec["values"]
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(
                f"sweep.params.{param_name}.values must be a non-empty list."
            )
        return [cast_value(value, value_type) for value in raw_values]

    if value_type == "bool":
        return [False, True]

    if value_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if step <= 0:
            raise ValueError(f"step must be > 0 for param {param_name!r}.")
        return list(range(low, high + 1, step))

    if value_type == "float" and {"low", "high", "step"} <= set(spec):
        low = float(spec["low"])
        high = float(spec["high"])
        step = float(spec["step"])
        if step <= 0:
            raise ValueError(f"step must be > 0 for param {param_name!r}.")
        values: list[float] = []
        current = low
        while current <= high + 1e-12:
            values.append(round(current, 12))
            current += step
        if not values:
            raise ValueError(f"Could not build float grid for {param_name!r}.")
        return values

    raise ValueError(
        f"Grid search needs values/expandable range for param {param_name!r}."
    )


def sample_random_value(
    rng: random.Random, param_name: str, spec: dict[str, Any]
) -> Any:
    value_type = str(spec["type"])
    if "values" in spec:
        candidates = spec_values_for_grid(param_name, spec)
        return rng.choice(candidates)

    if value_type == "bool":
        return rng.choice([False, True])

    if value_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if step <= 0:
            raise ValueError(f"step must be > 0 for param {param_name!r}.")
        candidates = list(range(low, high + 1, step))
        if not candidates:
            raise ValueError(f"Empty integer range for param {param_name!r}.")
        return rng.choice(candidates)

    if value_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if high < low:
            raise ValueError(f"high must be >= low for param {param_name!r}.")
        if as_bool(spec.get("log", False)):
            if low <= 0 or high <= 0:
                raise ValueError(
                    f"log float range for {param_name!r} requires low/high > 0."
                )
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)

    raise ValueError(f"Random search cannot sample param {param_name!r}.")


def suggest_optuna_value(trial: Any, param_name: str, spec: dict[str, Any]) -> Any:
    value_type = str(spec["type"])
    if "values" in spec:
        candidates = spec_values_for_grid(param_name, spec)
        return trial.suggest_categorical(param_name, candidates)

    if value_type == "bool":
        return trial.suggest_categorical(param_name, [False, True])
    if value_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = as_bool(spec.get("log", False))
        return trial.suggest_int(param_name, low, high, step=step, log=log)
    if value_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        log = as_bool(spec.get("log", False))
        step = spec.get("step")
        if step is None:
            return trial.suggest_float(param_name, low, high, log=log)
        return trial.suggest_float(param_name, low, high, step=float(step), log=log)

    raise ValueError(f"Optuna cannot suggest param {param_name!r}.")


def scalarize(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalarize(row.get(key)) for key in fieldnames})


def execute_trial(
    *,
    trial_index: int,
    trial_origin: str,
    trial_params: dict[str, Any],
    train_module: str,
    base_train_args: dict[str, Any],
    sweep_dir: Path,
    metric_cfg: MetricConfig,
    dry_run: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    label = run_label(trial_params)
    run_id = f"run_{trial_index:03d}_{label}"
    run_dir = sweep_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_args = dict(base_train_args)
    train_args.update(trial_params)
    train_args["output_dir"] = str(run_dir)
    if "save_all_checkpoints" not in train_args:
        train_args["save_all_checkpoints"] = True

    too_many_params, estimated_params = precheck_max_parameters(train_args)
    trial_meta: dict[str, Any] = {
        "trial_index": trial_index,
        "trial_origin": trial_origin,
        "run_id": run_id,
        "run_dir": str(run_dir),
        **trial_params,
    }
    if estimated_params is not None:
        trial_meta["num_parameters"] = estimated_params
        trial_meta["num_parameters_millions"] = round(estimated_params / 1_000_000, 3)

    if too_many_params:
        max_params = train_args.get("max_parameters")
        return {
            "status": "skipped",
            "trial_meta": trial_meta,
            "reason": "too_many_parameters",
            "max_parameters": max_params,
            "objective_value": None,
            "checkpoint_rows": [],
            "run_row": None,
        }

    cmd = build_train_command(train_module=train_module, train_args=train_args)
    print(f"[sweep] {run_id} params={json.dumps(trial_params, ensure_ascii=False)}")

    if dry_run:
        return {
            "status": "planned",
            "trial_meta": trial_meta,
            "reason": "dry_run",
            "objective_value": None,
            "checkpoint_rows": [],
            "run_row": None,
        }

    try:
        subprocess.run(cmd, check=True)
        history_path = run_dir / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")
        with history_path.open("r", encoding="utf-8") as fp:
            history = json.load(fp)
        if not isinstance(history, list):
            raise ValueError(f"Unexpected history format in {history_path}")
    except Exception as exc:
        if fail_fast:
            raise
        return {
            "status": "failed",
            "trial_meta": trial_meta,
            "reason": str(exc),
            "objective_value": None,
            "checkpoint_rows": [],
            "run_row": None,
        }

    best_epoch_row: dict[str, Any] | None = None
    best_objective: float | None = None
    checkpoint_rows: list[dict[str, Any]] = []
    for epoch_metrics in history:
        if not isinstance(epoch_metrics, dict):
            continue
        epoch = int(epoch_metrics.get("epoch", 0))
        checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        objective_value = metric_value(epoch_metrics, metric_cfg)
        row: dict[str, Any] = {
            **trial_meta,
            "checkpoint": str(checkpoint_path),
            "epoch": epoch,
            "objective_metric": metric_cfg.key,
            "objective_mode": metric_cfg.mode,
            "objective_value": objective_value,
        }
        for key, value in epoch_metrics.items():
            if key == "epoch":
                continue
            row[key] = value
        checkpoint_rows.append(row)

        if is_better(objective_value, best_objective, metric_cfg.mode):
            best_objective = objective_value
            best_epoch_row = row

    if best_epoch_row is None:
        return {
            "status": "failed",
            "trial_meta": trial_meta,
            "reason": "No epoch metrics found in history.json",
            "objective_value": None,
            "checkpoint_rows": [],
            "run_row": None,
        }

    run_row = {
        **trial_meta,
        "best_checkpoint": best_epoch_row["checkpoint"],
        "best_epoch": best_epoch_row["epoch"],
        "objective_metric": metric_cfg.key,
        "objective_mode": metric_cfg.mode,
        "objective_value": best_objective,
        "best_dev_primary_hmean_cer": best_epoch_row.get("dev_primary_hmean_cer"),
        "best_dev_cer": best_epoch_row.get("dev_cer"),
        "best_dev_in_domain_cer": best_epoch_row.get("dev_in_domain_cer"),
        "best_dev_out_of_domain_cer": best_epoch_row.get("dev_out_of_domain_cer"),
        "best_dev_loss": best_epoch_row.get("dev_loss"),
    }
    return {
        "status": "completed",
        "trial_meta": trial_meta,
        "reason": "",
        "objective_value": best_objective,
        "checkpoint_rows": checkpoint_rows,
        "run_row": run_row,
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    output_root = Path(config.get("output_root", "artifacts/sweeps"))
    run_name = str(config.get("run_name", "")).strip() or datetime.now().strftime(
        "baseline_sweep_%Y%m%d_%H%M%S"
    )
    sweep_dir = output_root.resolve() / run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = ensure_mapping(config.get("train", {}), key="train")
    train_module = str(train_cfg.get("module", "src.train_baseline"))
    base_train_args = ensure_mapping(train_cfg.get("args", {}), key="train.args")

    sweep_cfg = ensure_mapping(config.get("sweep", {}), key="sweep")
    method = normalize_method(sweep_cfg.get("method", "grid_search"))
    fail_fast = as_bool(sweep_cfg.get("fail_fast", True))
    space_specs = normalize_space_specs(sweep_cfg.get("params", {}))

    metric_cfg_map = ensure_mapping(sweep_cfg.get("metric", {}), key="sweep.metric")
    metric_cfg = MetricConfig(
        key=str(metric_cfg_map.get("key", "dev_primary_hmean_cer")),
        fallback_key=(
            str(metric_cfg_map["fallback_key"])
            if metric_cfg_map.get("fallback_key") is not None
            else "dev_cer"
        ),
        mode=str(metric_cfg_map.get("mode", "min")).strip().lower(),
    )
    if metric_cfg.mode not in {"min", "max"}:
        raise ValueError("sweep.metric.mode must be 'min' or 'max'.")

    checkpoint_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    planned_rows: list[dict[str, object]] = []
    run_index = 0

    def handle_trial(trial_params: dict[str, Any], *, trial_origin: str) -> float | None:
        nonlocal run_index
        run_index += 1
        result = execute_trial(
            trial_index=run_index,
            trial_origin=trial_origin,
            trial_params=trial_params,
            train_module=train_module,
            base_train_args=base_train_args,
            sweep_dir=sweep_dir,
            metric_cfg=metric_cfg,
            dry_run=args.dry_run,
            fail_fast=fail_fast,
        )
        status = str(result["status"])
        trial_meta = dict(result["trial_meta"])
        if status == "completed":
            checkpoint_rows.extend(result["checkpoint_rows"])
            run_row = result["run_row"]
            assert isinstance(run_row, dict)
            run_rows.append(run_row)
            return float(result["objective_value"]) if result["objective_value"] is not None else None
        if status == "planned":
            planned_rows.append({**trial_meta, "reason": result["reason"]})
            return None
        if status == "skipped":
            skipped_rows.append(
                {
                    **trial_meta,
                    "reason": result["reason"],
                    "max_parameters": result.get("max_parameters"),
                }
            )
            return None
        failed_rows.append({**trial_meta, "reason": result["reason"]})
        return None

    if method == "grid_search":
        names = list(space_specs.keys())
        grid_values = [spec_values_for_grid(name, space_specs[name]) for name in names]
        for combo in itertools.product(*grid_values):
            trial_params = {name: value for name, value in zip(names, combo)}
            handle_trial(trial_params, trial_origin="grid_search")
    elif method == "random_search":
        random_cfg = ensure_mapping(
            sweep_cfg.get("random_search", {}), key="sweep.random_search"
        )
        n_trials = int(random_cfg.get("n_trials", 20))
        seed = int(random_cfg.get("seed", 42))
        rng = random.Random(seed)
        for _ in range(n_trials):
            trial_params = {
                name: sample_random_value(rng, name, spec)
                for name, spec in space_specs.items()
            }
            handle_trial(trial_params, trial_origin="random_search")
    else:
        optuna_cfg = ensure_mapping(sweep_cfg.get("optuna", {}), key="sweep.optuna")
        n_trials = int(optuna_cfg.get("n_trials", 30))
        seed = int(optuna_cfg.get("seed", 42))
        direction = "minimize" if metric_cfg.mode == "min" else "maximize"
        if args.dry_run:
            for idx in range(1, n_trials + 1):
                planned_rows.append(
                    {
                        "trial_index": idx,
                        "trial_origin": "optuna",
                        "run_id": f"planned_optuna_{idx:03d}",
                        "reason": "dry_run",
                    }
                )
        else:
            try:
                optuna_module: Any = importlib.import_module("optuna")
            except ImportError as exc:
                raise RuntimeError(
                    "Optuna is not installed. Install it with `uv add optuna` "
                    "or switch sweep.method to grid_search/random_search."
                ) from exc

            sampler_name = str(optuna_cfg.get("sampler", "tpe")).strip().lower()
            if sampler_name == "random":
                sampler = optuna_module.samplers.RandomSampler(seed=seed)
            else:
                sampler = optuna_module.samplers.TPESampler(seed=seed)

            study = optuna_module.create_study(
                direction=direction,
                sampler=sampler,
                study_name=optuna_cfg.get("study_name"),
                storage=optuna_cfg.get("storage"),
                load_if_exists=as_bool(optuna_cfg.get("load_if_exists", False)),
            )

            penalty = float("inf") if metric_cfg.mode == "min" else float("-inf")

            def objective(trial: Any) -> float:
                trial_params = {
                    name: suggest_optuna_value(trial, name, spec)
                    for name, spec in space_specs.items()
                }
                objective_value = handle_trial(
                    trial_params, trial_origin=f"optuna_trial_{trial.number}"
                )
                if objective_value is None:
                    return penalty
                return objective_value

            study.optimize(objective, n_trials=n_trials)

    checkpoint_rows.sort(
        key=lambda row: metric_sort_key(
            float(row["objective_value"]) if row.get("objective_value") is not None else None,
            metric_cfg.mode,
        )
    )
    run_rows.sort(
        key=lambda row: metric_sort_key(
            float(row["objective_value"]) if row.get("objective_value") is not None else None,
            metric_cfg.mode,
        )
    )

    for idx, row in enumerate(checkpoint_rows, start=1):
        row["rank"] = idx
    for idx, row in enumerate(run_rows, start=1):
        row["rank"] = idx

    checkpoints_csv = sweep_dir / "checkpoint_metrics_dev.csv"
    runs_csv = sweep_dir / "run_best_dev.csv"
    skipped_csv = sweep_dir / "skipped_by_param_limit.csv"
    failed_csv = sweep_dir / "failed_trials.csv"
    planned_csv = sweep_dir / "planned_trials.csv"
    summary_json = sweep_dir / "sweep_summary.json"

    write_csv(checkpoints_csv, checkpoint_rows)
    write_csv(runs_csv, run_rows)
    write_csv(skipped_csv, skipped_rows)
    write_csv(failed_csv, failed_rows)
    write_csv(planned_csv, planned_rows)

    best_checkpoint = run_rows[0]["best_checkpoint"] if run_rows else None
    summary = {
        "sweep_dir": str(sweep_dir),
        "method": method,
        "train_module": train_module,
        "metric": {
            "key": metric_cfg.key,
            "fallback_key": metric_cfg.fallback_key,
            "mode": metric_cfg.mode,
        },
        "num_runs_finished": len(run_rows),
        "num_checkpoint_rows": len(checkpoint_rows),
        "num_skipped_by_param_limit": len(skipped_rows),
        "num_failed_trials": len(failed_rows),
        "num_planned_trials": len(planned_rows),
        "best_checkpoint": best_checkpoint,
        "config_path": str(config_path),
        "tables": {
            "checkpoint_metrics_dev_csv": str(checkpoints_csv),
            "run_best_dev_csv": str(runs_csv),
            "skipped_by_param_limit_csv": str(skipped_csv),
            "failed_trials_csv": str(failed_csv),
            "planned_trials_csv": str(planned_csv),
        },
    }
    with summary_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

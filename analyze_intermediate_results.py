#!/usr/bin/env python3
"""Read-only analytics for a running interaction/stability-selection pipeline.

The script never changes pipeline checkpoints, caches or results.  It reads the
atomic ``search_state.json`` checkpoint and append-only resource log, then writes
a separate intermediate report with CSV tables and optional PNG charts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DEFAULT_STABILITY_DIR = Path("/home/datalab/nfs/Пробник/stability_selection")


def read_json(path: Path, retries: int = 3) -> Any:
    """Read an atomically replaced JSON file, retrying transient races."""
    error: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.15)
    raise RuntimeError(f"Cannot read valid JSON from {path}: {error}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete rows from an append-only JSONL, ignoring a partial tail."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, tuple, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def discover_run(stability_dir: Path, explicit_run: Path | None) -> Path:
    if explicit_run is not None:
        run = explicit_run.expanduser().resolve()
        if not (run / "search_state.json").is_file():
            raise FileNotFoundError(f"No search_state.json in {run}")
        return run

    states = list(stability_dir.glob("fold_runs/*/search_state.json"))
    final_state = stability_dir / "final_run" / "search_state.json"
    if final_state.is_file():
        states.append(final_state)
    if not states:
        raise FileNotFoundError(
            f"No running/completed search_state.json under {stability_dir}"
        )
    return max(states, key=lambda item: item.stat().st_mtime).parent


def candidate_row(item: dict[str, Any], sequence: int) -> dict[str, Any]:
    metrics = item.get("metrics") or {}
    subset = list(item.get("subset") or [])
    return {
        "sequence": sequence,
        "fidelity": item.get("fidelity"),
        "n_features": item.get("n_features", len(subset)),
        "valid_metric": safe_float(item.get("valid_metric")),
        "train_metric": safe_float(item.get("train_metric")),
        "primary_metric_gap": safe_float(item.get("gap")),
        "metric_std": safe_float(item.get("metric_std")),
        "constraint_violation": safe_float(
            item.get("constraint_violation", 0.0)
        ),
        "train_gini": safe_float(metrics.get("train_gini")),
        "valid_gini": safe_float(metrics.get("valid_gini")),
        "gini_gap_diagnostic": safe_float(metrics.get("gini_gap")),
        "runtime_seconds": safe_float(item.get("runtime_seconds")),
        "features": subset,
    }


def candidate_rank(row: dict[str, Any]) -> tuple[float, float, int, float]:
    violation = row["constraint_violation"]
    valid = row["valid_metric"]
    count = int(row["n_features"] or 0)
    std = row["metric_std"]
    return (
        violation if violation is not None else float("inf"),
        -(valid if valid is not None else -float("inf")),
        count,
        std if std is not None else float("inf"),
    )


def deduplicate_candidates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["fidelity"]), tuple(sorted(row["features"])))
        if key not in best or candidate_rank(row) < candidate_rank(best[key]):
            best[key] = row
    return list(best.values())


def approximate_pareto(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [
        row
        for row in rows
        if (row["constraint_violation"] or 0.0) <= 1e-12
        and row["valid_metric"] is not None
    ]
    result: list[dict[str, Any]] = []
    for row in feasible:
        dominated = any(
            other is not row
            and other["valid_metric"] >= row["valid_metric"]
            and int(other["n_features"]) <= int(row["n_features"])
            and (
                other["valid_metric"] > row["valid_metric"]
                or int(other["n_features"]) < int(row["n_features"])
            )
            for other in feasible
        )
        if not dominated:
            result.append(row)
    return sorted(result, key=lambda row: (int(row["n_features"]), -row["valid_metric"]))


def factor_tables(
    top_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_count: Counter[str] = Counter()
    pair_count: Counter[tuple[str, str]] = Counter()
    for row in top_rows:
        features = sorted(set(row["features"]))
        feature_count.update(features)
        pair_count.update(itertools.combinations(features, 2))
    denominator = max(1, len(top_rows))
    feature_rows = [
        {
            "feature": feature,
            "count_in_top": count,
            "frequency_in_top": count / denominator,
        }
        for feature, count in feature_count.most_common()
    ]
    pair_rows = [
        {
            "first": pair[0],
            "second": pair[1],
            "count_in_top": count,
            "frequency_in_top": count / denominator,
        }
        for pair, count in pair_count.most_common()
    ]
    return feature_rows, pair_rows


def completed_fold_tables(
    stability_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in sorted(stability_dir.glob("fold_runs/*/selected_features.json")):
        try:
            result = read_json(path)
        except RuntimeError:
            continue
        features = list(result.get("features") or [])
        counts.update(features)
        folds.append(
            {
                "fold": path.parent.name,
                "n_features": result.get("n_features", len(features)),
                "valid_metric": result.get("valid_metric"),
                "metric_std": result.get("metric_std"),
                "primary_metric_gap": result.get("gap"),
                "selection_basis": result.get("selection_basis"),
                "features": features,
            }
        )
    denominator = max(1, len(folds))
    stability = [
        {
            "feature": feature,
            "completed_folds": count,
            "frequency_among_completed_folds": count / denominator,
        }
        for feature, count in counts.most_common()
    ]
    return folds, stability


def resource_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    gpu_free: dict[str, list[float]] = {}
    for row in rows:
        for gpu in row.get("gpus") or []:
            free = safe_float(gpu.get("free_gb"))
            if free is not None:
                gpu_free.setdefault(str(gpu.get("device")), []).append(free)
    return {
        "observations": len(rows),
        "max_active_trials": max(int(row.get("active_trials", 0)) for row in rows),
        "max_pending_trials": max(int(row.get("pending_trials", 0)) for row in rows),
        "max_cpu_percent": max(
            (safe_float(row.get("cpu_percent")) or 0.0) for row in rows
        ),
        "max_ram_percent": max(
            (safe_float(row.get("ram_percent")) or 0.0) for row in rows
        ),
        "min_ram_available_gb": min(
            (safe_float(row.get("ram_available_gb")) or float("inf"))
            for row in rows
        ),
        "max_process_rss_gb": max(
            (safe_float(row.get("process_rss_gb")) or 0.0) for row in rows
        ),
        "zero_allowed_with_pending": sum(
            1
            for row in rows
            if int(row.get("allowed_trials", 0)) == 0
            and int(row.get("pending_trials", 0)) > 0
        ),
        "minimum_gpu_free_gb": {
            device: min(values) for device, values in gpu_free.items()
        },
    }


def diagnostic_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "failed_trials": len(rows),
        "oom_trials": sum(bool(row.get("oom_detected")) for row in rows),
        "termination_reasons": dict(
            Counter(str(row.get("termination_reason") or "none") for row in rows)
        ),
        "error_types": dict(
            Counter(str(row.get("error_type") or "unknown") for row in rows)
        ),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    progress = summary["progress"]
    resources = summary["resources"]
    diagnostics = summary["diagnostics"]
    top = summary["top_candidates"]
    features = summary["top_features"]
    folds = summary["completed_folds"]
    stability = summary["fold_feature_stability"]
    lines = [
        "# Промежуточный отчёт по отбору факторов",
        "",
        f"Сформирован: `{summary['generated_at']}`",
        f"Текущий каталог: `{summary['run_directory']}`",
        "",
        "## Прогресс",
        "",
        f"- Статус checkpoint: `{progress['status']}`",
        f"- Этап: `{progress['stage']}`",
        f"- Рестарт: `{progress['restart']}`",
        f"- Следующее поколение: `{progress['next_generation']}`",
        f"- Полных уникальных оценок: **{progress['unique_full_trials']}**",
        f"- Скрининговых оценок: **{progress['screening_trials']}**",
        f"- Батчей продвижения: **{progress['promotion_batches']}**",
        f"- Завершённых рестартов: **{progress['completed_restarts']}**",
        "",
        "## Лучшие текущие комбинации",
        "",
        "| № | valid | train | primary gap | GINI gap (диагн.) | std | факторов | violation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(top[:15], 1):
        lines.append(
            f"| {index} | {fmt(row['valid_metric'])} | {fmt(row['train_metric'])} "
            f"| {fmt(row['primary_metric_gap'])} | {fmt(row['gini_gap_diagnostic'])} "
            f"| {fmt(row['metric_std'])} | {row['n_features']} "
            f"| {fmt(row['constraint_violation'])} |"
        )
    lines.extend(["", "## Наиболее частые факторы в top", ""])
    lines.extend(
        f"- `{row['feature']}` — {row['frequency_in_top']:.1%} "
        f"({row['count_in_top']} комбинаций)"
        for row in features[:25]
    )
    lines.extend(
        [
            "",
            "## Ресурсы и ошибки",
            "",
            f"- Максимум одновременных trials: **{resources.get('max_active_trials', '—')}**",
            f"- Максимальная загрузка CPU: **{fmt(resources.get('max_cpu_percent'), 2)}%**",
            f"- Максимальная загрузка RAM: **{fmt(resources.get('max_ram_percent'), 2)}%**",
            f"- Минимум свободной RAM: **{fmt(resources.get('min_ram_available_gb'), 2)} GB**",
            f"- Максимальный RSS процесса: **{fmt(resources.get('max_process_rss_gb'), 2)} GB**",
            f"- Наблюдений с pending>0 и allowed=0: **{resources.get('zero_allowed_with_pending', 0)}**",
            f"- Неуспешных trials: **{diagnostics.get('failed_trials', 0)}**",
            f"- OOM trials: **{diagnostics.get('oom_trials', 0)}**",
            "",
            "## Завершённые фолды",
            "",
            f"Завершено фолдов: **{len(folds)}**. Частоты до завершения всех фолдов предварительные.",
            "",
        ]
    )
    for row in folds:
        lines.append(
            f"- `{row['fold']}`: {row['n_features']} факторов, "
            f"valid={fmt(row['valid_metric'])}"
        )
    lines.extend(["", "### Частота факторов по завершённым фолдам", ""])
    lines.extend(
        f"- `{row['feature']}` — {row['frequency_among_completed_folds']:.1%} "
        f"({row['completed_folds']}/{len(folds)})"
        for row in stability[:25]
    )
    lines.extend(
        [
            "",
            "## Как интерпретировать",
            "",
            "- Текущий лидер не является финальным победителем до confirm/OOS и завершения stability selection.",
            "- Сравнивать screen и search напрямую нельзя: это разные fidelity и объёмы обучения.",
            "- При близкой valid-метрике предпочтительнее компактный набор с меньшим metric_std.",
            "- GINI gap в версии 0.13.0 — только диагностика и не ограничивает отбор.",
            "- Частота по одному завершённому фолду всегда равна 100% и ничего не говорит об устойчивости.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_plots(
    output: Path,
    full_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    resources: Sequence[dict[str, Any]],
    fold_stability: Sequence[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib is not installed; PNG charts were skipped"]

    notes: list[str] = []
    valid = [row["valid_metric"] for row in full_rows]
    valid = [value for value in valid if value is not None]
    if valid:
        running_best: list[float] = []
        current = -float("inf")
        for value in valid:
            current = max(current, value)
            running_best.append(current)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(range(1, len(valid) + 1), valid, alpha=0.30, label="trial")
        ax.plot(range(1, len(valid) + 1), running_best, linewidth=2, label="best so far")
        ax.set(xlabel="Полная оценка", ylabel="Valid metric", title="Сходимость поиска")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "01_convergence.png", dpi=160)
        plt.close(fig)

    if feature_rows:
        data = feature_rows[:25][::-1]
        fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.30)))
        ax.barh(
            [row["feature"] for row in data],
            [row["frequency_in_top"] for row in data],
        )
        ax.set(xlabel="Частота в top", title="Факторы в лучших текущих комбинациях")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "02_top_features.png", dpi=160)
        plt.close(fig)

    if resources:
        x = range(1, len(resources) + 1)
        cpu = [safe_float(row.get("cpu_percent")) or 0.0 for row in resources]
        ram = [safe_float(row.get("ram_percent")) or 0.0 for row in resources]
        active = [int(row.get("active_trials", 0)) for row in resources]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(x, cpu, label="CPU, %")
        ax1.plot(x, ram, label="RAM, %")
        ax1.set_ylabel("Использование, %")
        ax1.grid(alpha=0.25)
        ax1.legend()
        ax2.step(x, active, where="post", label="active trials")
        ax2.set(xlabel="Снимок ресурсов", ylabel="Trials")
        ax2.grid(alpha=0.25)
        ax2.legend()
        fig.suptitle("Динамика ресурсов")
        fig.tight_layout()
        fig.savefig(output / "03_resources.png", dpi=160)
        plt.close(fig)

    if fold_stability:
        data = fold_stability[:25][::-1]
        fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.30)))
        ax.barh(
            [row["feature"] for row in data],
            [row["frequency_among_completed_folds"] for row in data],
        )
        ax.set(xlabel="Частота", title="Устойчивость по завершённым фолдам")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "04_fold_stability.png", dpi=160)
        plt.close(fig)
    return notes


def analyze_once(args: argparse.Namespace) -> dict[str, Any]:
    stability_dir = args.stability_dir.expanduser().resolve()
    run_dir = discover_run(stability_dir, args.run_dir)
    state_path = run_dir / "search_state.json"
    state = read_json(state_path)

    full_rows = [
        candidate_row(item, index)
        for index, item in enumerate(state.get("history") or [], 1)
    ]
    screening_rows = [
        candidate_row(item, index)
        for index, item in enumerate(state.get("screening_history") or [], 1)
    ]
    unique_full = deduplicate_candidates(full_rows)
    ranked = sorted(unique_full, key=candidate_rank)
    feasible_ranked = [
        row
        for row in ranked
        if (row["constraint_violation"] or 0.0) <= 1e-12
    ]
    top = (feasible_ranked or ranked)[: args.top]
    pareto = approximate_pareto(unique_full)
    feature_rows, pair_rows = factor_tables(top)

    coalitions = sorted(
        state.get("coalitions") or [],
        key=lambda row: (
            -(safe_float(row.get("score")) or 0.0),
            -int(row.get("support", 0)),
        ),
    )
    diagnostics = list(state.get("trial_diagnostics") or [])
    resource_rows = read_jsonl(run_dir / "resource_usage.jsonl")
    fold_rows, fold_stability = completed_fold_tables(stability_dir)
    active = state.get("active") or {}
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stability_directory": str(stability_dir),
        "run_directory": str(run_dir),
        "checkpoint_mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(state_path.stat().st_mtime),
        ),
        "progress": {
            "status": state.get("status"),
            "stage": active.get("stage") or "between_restarts_or_complete",
            "restart": active.get("restart"),
            "next_generation": active.get("next_generation"),
            "full_trials": len(full_rows),
            "unique_full_trials": len(unique_full),
            "screening_trials": len(screening_rows),
            "promotion_batches": len(state.get("promotion_batches") or []),
            "completed_restarts": len(state.get("completed_restart_fronts") or []),
            "checkpoint_has_final_front": state.get("final_front") is not None,
        },
        "best_valid_metric": max(
            (row["valid_metric"] for row in unique_full if row["valid_metric"] is not None),
            default=None,
        ),
        "feasible_unique_candidates": len(feasible_ranked),
        "approximate_pareto_size": len(pareto),
        "top_candidates": top,
        "approximate_pareto": pareto,
        "top_features": feature_rows,
        "top_pairs": pair_rows,
        "coalitions": coalitions[: args.top],
        "diagnostics": diagnostic_summary(diagnostics),
        "resources": resource_summary(resource_rows),
        "completed_folds": fold_rows,
        "fold_feature_stability": fold_stability,
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "intermediate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "intermediate_report.md").write_text(
        build_markdown(summary), encoding="utf-8"
    )
    write_csv(output / "top_candidates.csv", top)
    write_csv(output / "approximate_pareto.csv", pareto)
    write_csv(output / "top_features.csv", feature_rows)
    write_csv(output / "top_pairs.csv", pair_rows)
    write_csv(output / "coalitions.csv", coalitions)
    write_csv(output / "trial_diagnostics.csv", diagnostics)
    write_csv(output / "completed_folds.csv", fold_rows)
    write_csv(output / "fold_feature_stability.csv", fold_stability)
    write_csv(output / "resource_usage_tail.csv", resource_rows[-args.resource_tail :])
    if not args.no_plots:
        summary["plot_notes"] = make_plots(
            output, full_rows, feature_rows, resource_rows, fold_stability
        )
        (output / "intermediate_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("=" * 78)
    print("ПРОМЕЖУТОЧНАЯ АНАЛИТИКА")
    print("=" * 78)
    print(f"Текущий прогон:       {run_dir}")
    print(f"Checkpoint:           {summary['checkpoint_mtime']}")
    print(f"Статус / этап:        {summary['progress']['status']} / {summary['progress']['stage']}")
    print(f"Уникальных trials:    {len(unique_full)}")
    print(f"Feasible trials:      {len(feasible_ranked)}")
    print(f"Лучший valid:         {fmt(summary['best_valid_metric'])}")
    print(f"Approx Pareto:        {len(pareto)}")
    print(f"Завершено фолдов:     {len(fold_rows)}")
    print(f"Ошибок / OOM:         {len(diagnostics)} / {summary['diagnostics']['oom_trials']}")
    print(f"Отчёт:                {output / 'intermediate_report.md'}")
    print(f"Таблицы и графики:    {output}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stability-dir",
        type=Path,
        default=DEFAULT_STABILITY_DIR,
        help="Root output_directory of run_stability_selection.py",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Analyze this exact selector output instead of auto-discovery",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("intermediate_analysis"),
        help="Separate directory for generated analytics",
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--resource-tail", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh repeatedly with this interval; stop with Kernel Interrupt",
    )
    args = parser.parse_args(argv)
    if args.top < 1 or args.resource_tail < 1:
        parser.error("--top and --resource-tail must be positive")
    if args.watch_seconds < 0:
        parser.error("--watch-seconds cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        try:
            analyze_once(args)
        except FileNotFoundError as exc:
            print(f"Данных для анализа пока нет: {exc}")
        if args.watch_seconds <= 0:
            return 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

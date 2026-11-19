"""Small command-line interface for agents and shell pipelines.

``pymaftools <command> --help`` documents each command. Every command can emit
JSON (``--json``) that carries the same provenance as the Python API, so an
MCP server or agent can wrap these commands without parsing prose::

    pymaftools inspect cohort.maf --json
    pymaftools oncoplot cohort.maf -o oncoplot.png --top 20
    pymaftools interactions cohort.maf --top 25 --json
    pymaftools session-info

Exit codes: 0 success; 1 error (message on stderr, or a JSON ``error`` object
with ``--json``); 2 input rejected by ``--validate strict``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2


def _read_maf(path: str, validate: str, all_classes: bool):
    from .core.MAF import MAF

    maf = MAF.read_maf(path, validate=validate)
    if not all_classes:
        maf = maf.filter_maf(MAF.nonsynonymous_types)
    return maf


def _gene_table(maf):
    return maf.to_gene_table()


def _top_genes(table, top: int) -> list[dict[str, Any]]:
    import pandas as pd

    binary = pd.DataFrame(table.to_binary_table()).astype(bool)
    counts = binary.sum(axis=1).sort_values(ascending=False, kind="stable").head(top)
    n_samples = binary.shape[1]
    return [
        {
            "gene": str(gene),
            "mutated_samples": int(count),
            "fraction": (float(count) / n_samples) if n_samples else 0.0,
        }
        for gene, count in counts.items()
    ]


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    from .core.MAF import MAF
    from .provenance import session_info

    full = MAF.read_maf(args.maf, validate=args.validate)
    report = full.ingest_report
    classes = full["Variant_Classification"].value_counts()
    nonsynonymous = full.filter_maf(MAF.nonsynonymous_types)
    table = _gene_table(nonsynonymous)
    return {
        "command": "inspect",
        "ingest_report": report.to_dict() if report is not None else None,
        "n_events": int(len(full)),
        "n_samples": int(full["sample_ID"].nunique()),
        "n_genes": int(full["Hugo_Symbol"].nunique()),
        "n_nonsynonymous_events": int(len(nonsynonymous)),
        "variant_classification_counts": {
            str(name): int(count) for name, count in classes.items()
        },
        "top_genes_nonsynonymous": _top_genes(table, args.top),
        "session": session_info(),
    }


def cmd_oncoplot(args: argparse.Namespace) -> dict[str, Any]:
    from .plot.summary import write_figure_summary
    from .provenance import Provenance, table_inputs, table_sample_universe

    maf = _read_maf(args.maf, args.validate, args.all_classes)
    table = _gene_table(maf)
    if table.shape[0] == 0:
        raise ValueError(
            "No genes left to plot after filtering"
            + ("" if args.all_classes else " to nonsynonymous variant classes")
            + "; pass --all-classes to keep every classification."
        )
    table = table.add_freq().sort_features(by="freq").head(args.top)
    oncoplot = table.plot.oncoplot(figsize=(args.width, args.height)).main()
    oncoplot = oncoplot.add_freq().add_bar("mutations_count", side="top").render()
    oncoplot.save(args.output, dpi=args.dpi)
    record = oncoplot.summary(dpi=args.dpi)
    record["provenance"] = Provenance(
        analysis="cli.oncoplot",
        parameters={
            "top": args.top,
            "all_classes": args.all_classes,
            "validate": args.validate,
            "figsize": [args.width, args.height],
            "dpi": args.dpi,
        },
        inputs=table_inputs(table),
        sample_universe=table_sample_universe(table),
    ).to_dict()
    sidecar = write_figure_summary(record, args.output, figure_path=args.output)
    oncoplot.close()
    record["command"] = "oncoplot"
    record["summary_path"] = str(sidecar)
    return record


def cmd_interactions(args: argparse.Namespace) -> dict[str, Any]:
    from .plot.wes import somatic_interactions
    from .provenance import dataframe_to_dict

    maf = _read_maf(args.maf, args.validate, args.all_classes)
    table = _gene_table(maf)
    stats = somatic_interactions(table, top=args.top, alpha=args.alpha)
    if args.output:
        if args.output.endswith(".json"):
            from .provenance import to_json

            to_json(stats, args.output)
        else:
            stats.to_csv(
                args.output,
                sep="\t" if args.output.endswith(".tsv") else ",",
                index=False,
            )
    record = dataframe_to_dict(stats)
    record["command"] = "interactions"
    record["n_significant"] = (
        int(stats["is_significant"].sum()) if "is_significant" in stats else 0
    )
    if args.output:
        record["output"] = args.output
    return record


def cmd_session_info(args: argparse.Namespace) -> dict[str, Any]:
    from .provenance import session_info

    return {"command": "session-info", **session_info()}


# --------------------------------------------------------------------------
# text rendering (non-JSON mode)
# --------------------------------------------------------------------------


def _print_text(record: dict[str, Any]) -> None:
    command = record.get("command")
    if command == "inspect":
        report = record.get("ingest_report") or {}
        print(
            f"{report.get('path')}: {record['n_events']} events, "
            f"{record['n_samples']} samples, {record['n_genes']} genes "
            f"({record['n_nonsynonymous_events']} nonsynonymous events)"
        )
        print(
            f"  source guess: {report.get('source')}; "
            f"genome build: {', '.join(report.get('genome_builds') or []) or 'unknown'}"
        )
        for issue in report.get("issues") or []:
            print(f"  ISSUE: {issue}")
        for note in report.get("notes") or []:
            print(f"  note: {note}")
        print(
            "  top genes (nonsynonymous): "
            + ", ".join(
                f"{g['gene']} {g['mutated_samples']}"
                for g in record["top_genes_nonsynonymous"]
            )
        )
    elif command == "oncoplot":
        print(
            f"{record['n_features']} genes x {record['n_samples']} samples, "
            f"fill {record['fill_fraction']:.1%}; summary: {record['summary_path']}"
        )
    elif command == "interactions":
        print(
            f"{record['n_rows']} gene pairs tested, "
            f"{record['n_significant']} significant"
        )
        for row in record["records"][:10]:
            print(
                f"  {row['gene1']} - {row['gene2']}: {row['interaction']}, "
                f"FDR={row['adjusted_p_value']}"
            )
    elif command == "session-info":
        print(
            f"pymaftools {record['pymaftools']} (Python {record['python']}, "
            f"{record['platform']})"
        )
        for name, value in record["dependencies"].items():
            print(f"  {name} {value}")
    for message in record.get("warnings") or []:
        print(f"  warning: {message}")
    for message in record.get("python_warnings") or []:
        print(f"  warning: {message}", file=sys.stderr)


# --------------------------------------------------------------------------
# parser / entry point
# --------------------------------------------------------------------------


def _add_maf_arguments(parser: argparse.ArgumentParser, validate_default: str) -> None:
    parser.add_argument("maf", help="MAF file (.maf, .maf.gz, tab or comma separated)")
    parser.add_argument(
        "--validate",
        choices=("strict", "warn", "off"),
        default=validate_default,
        help=f"read_maf validation mode (default: {validate_default})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymaftools",
        description="Inspect MAF files and draw/compute standard results.",
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON on stdout"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="ingest report and cohort overview")
    _add_maf_arguments(inspect, validate_default="warn")
    inspect.add_argument("--top", type=int, default=10, help="genes to list")
    inspect.set_defaults(func=cmd_inspect)

    onco = sub.add_parser("oncoplot", help="draw an oncoplot (+ JSON summary)")
    _add_maf_arguments(onco, validate_default="strict")
    onco.add_argument(
        "-o", "--output", required=True, help="figure path (.png/.pdf/.svg)"
    )
    onco.add_argument("--top", type=int, default=20, help="most mutated genes to plot")
    onco.add_argument(
        "--all-classes",
        action="store_true",
        help="keep every Variant_Classification (default: nonsynonymous)",
    )
    onco.add_argument("--width", type=float, default=12.0, help="inches")
    onco.add_argument("--height", type=float, default=8.0, help="inches")
    onco.add_argument("--dpi", type=int, default=200)
    onco.set_defaults(func=cmd_oncoplot)

    inter = sub.add_parser(
        "interactions", help="co-occurrence / mutual exclusivity statistics"
    )
    _add_maf_arguments(inter, validate_default="strict")
    inter.add_argument("--top", type=int, default=25, help="most mutated genes to test")
    inter.add_argument("--alpha", type=float, default=0.05, help="BH-FDR threshold")
    inter.add_argument(
        "--all-classes",
        action="store_true",
        help="keep every Variant_Classification (default: nonsynonymous)",
    )
    inter.add_argument(
        "-o", "--output", help="also write the table (.csv, .tsv or .json)"
    )
    inter.set_defaults(func=cmd_interactions)

    info = sub.add_parser("session-info", help="package and dependency versions")
    info.set_defaults(func=cmd_session_info)

    # Accept --json after the subcommand too (`pymaftools inspect x.maf --json`).
    for subparser in (inspect, onco, inter, info):
        subparser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="print machine-readable JSON on stdout",
        )
    return parser


def _run(func: Callable[[argparse.Namespace], dict], args) -> tuple[int, dict]:
    from .core.maf_io import MAFValidationError

    # Library progress prints go to stderr so stdout stays pure JSON.
    with (
        warnings.catch_warnings(record=True) as caught,
        contextlib.redirect_stdout(sys.stderr),
    ):
        warnings.simplefilter("always")
        try:
            record = func(args)
            code = EXIT_OK
        except MAFValidationError as error:
            record = {
                "command": args.command,
                "error": "MAFValidationError",
                "message": str(error),
                "ingest_report": error.report.to_dict(),
            }
            code = EXIT_INVALID_INPUT
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            record = {
                "command": args.command,
                "error": type(error).__name__,
                "message": str(error),
            }
            code = EXIT_ERROR
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    if messages:
        record["python_warnings"] = messages
    return code, record


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``pymaftools`` console script."""
    args = build_parser().parse_args(argv)
    code, record = _run(args.func, args)
    if args.json:
        from .provenance import to_json

        print(to_json(record))
    elif "error" in record:
        print(
            f"pymaftools {args.command}: {record['error']}: {record['message']}",
            file=sys.stderr,
        )
        for message in record.get("python_warnings") or []:
            print(f"  warning: {message}", file=sys.stderr)
    else:
        _print_text(record)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

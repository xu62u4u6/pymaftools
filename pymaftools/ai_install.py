"""Install the AI assets distributed with pymaftools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ASSET_ROOT = Path(__file__).with_name("ai_assets")


def _targets(target: str, scope: str, project_dir: Path) -> tuple[Path, Path]:
    root = Path.home() if scope == "user" else project_dir
    prefix = ".codex" if target == "codex" else ".claude"
    return root / prefix / "agents", root / prefix / "skills"


def _render_claude_agent(source: Path) -> str:
    with source.open("rb") as handle:
        agent = tomllib.load(handle)
    description = json.dumps(agent["description"], ensure_ascii=False)
    return (
        "---\n"
        f"name: {agent['name']}\n"
        f"description: {description}\n"
        "---\n\n"
        f"{agent['developer_instructions'].strip()}\n"
    )


def _write_text(destination: Path, content: str, *, dry_run: bool, force: bool) -> str:
    if destination.exists():
        if destination.read_text(encoding="utf-8") == content:
            return f"UNCHANGED {destination}"
        if not force:
            raise FileExistsError(
                f"{destination} already exists with different content; use --force"
            )
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return f"{'WOULD WRITE' if dry_run else 'WROTE'} {destination}"


def _copy_file(source: Path, destination: Path, *, dry_run: bool, force: bool) -> str:
    return _write_text(
        destination,
        source.read_text(encoding="utf-8"),
        dry_run=dry_run,
        force=force,
    )


def install(
    *,
    target: str,
    scope: str,
    components: set[str],
    project_dir: Path,
    dry_run: bool,
    force: bool,
) -> list[str]:
    agent_dir, skill_dir = _targets(target, scope, project_dir)
    actions: list[str] = []

    if "agents" in components:
        for source in sorted((ASSET_ROOT / "agents").glob("*.toml")):
            if target == "codex":
                actions.append(
                    _copy_file(
                        source,
                        agent_dir / source.name,
                        dry_run=dry_run,
                        force=force,
                    )
                )
            else:
                actions.append(
                    _write_text(
                        agent_dir / f"{source.stem}.md",
                        _render_claude_agent(source),
                        dry_run=dry_run,
                        force=force,
                    )
                )

    if "skill" in components:
        source_skill = ASSET_ROOT / "skills" / "pymaftools"
        for source in sorted(
            path for path in source_skill.rglob("*") if path.is_file()
        ):
            relative = source.relative_to(source_skill)
            actions.append(
                _copy_file(
                    source,
                    skill_dir / "pymaftools" / relative,
                    dry_run=dry_run,
                    force=force,
                )
            )

    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymaftools-ai",
        description="Install AI agents and the pymaftools skill.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List bundled AI assets.")

    install_parser = subparsers.add_parser("install", help="Install bundled assets.")
    install_parser.add_argument("--target", choices=("codex", "claude"), required=True)
    install_parser.add_argument("--scope", choices=("user", "project"), default="user")
    install_parser.add_argument(
        "--components",
        choices=("all", "agents", "skill"),
        default="all",
    )
    install_parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for agent in sorted((ASSET_ROOT / "agents").glob("*.toml")):
            print(f"agent {agent.stem}")
        print("skill pymaftools")
        return 0

    components = {"agents", "skill"} if args.components == "all" else {args.components}
    try:
        actions = install(
            target=args.target,
            scope=args.scope,
            components=components,
            project_dir=args.project_dir.resolve(),
            dry_run=args.dry_run,
            force=args.force,
        )
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 2

    for action in actions:
        print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

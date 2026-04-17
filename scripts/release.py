#!/usr/bin/env python3
"""Release automation helpers for Cyber-Inference."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT / "pyproject.toml"
README_PATH = ROOT / "README.md"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"

SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
PROJECT_VERSION_RE = re.compile(r'(?m)^(version = ")([^"]+)(")$')

BumpLevel = Literal["major", "minor", "patch"]


@dataclass(frozen=True)
class CommitEntry:
    sha: str
    subject: str
    body: str


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_project_version(pyproject_text: str | None = None) -> str:
    payload = pyproject_text if pyproject_text is not None else PYPROJECT_PATH.read_text(encoding="utf-8")
    data = tomllib.loads(payload)
    version = data["project"]["version"]
    if not isinstance(version, str):
        raise RuntimeError("Project version must be a string")
    return version


def get_latest_release_tag() -> str | None:
    try:
        tag = run_git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    except subprocess.CalledProcessError:
        return None
    return tag or None


def parse_commits(log_output: str) -> list[CommitEntry]:
    commits: list[CommitEntry] = []
    for raw_record in log_output.split("\x1e"):
        record = raw_record.strip("\n")
        if not record:
            continue
        fields = [field.strip() for field in record.split("\x1f", maxsplit=2)]
        while len(fields) < 3:
            fields.append("")
        sha, subject, body = fields
        commits.append(CommitEntry(sha=sha, subject=subject, body=body))
    return commits


def should_include_commit(subject: str) -> bool:
    lowered = subject.strip().lower()
    if not lowered:
        return False
    noisy_prefixes = (
        "merge ",
        "omx(",
        "chore(release):",
        "release ",
    )
    return not lowered.startswith(noisy_prefixes)


def get_relevant_commits(since_ref: str | None) -> list[CommitEntry]:
    command = ["log", "--no-merges", "--pretty=format:%H%x1f%s%x1f%b%x1e"]
    if since_ref:
        command.append(f"{since_ref}..HEAD")
    raw_log = run_git(*command)
    commits = parse_commits(raw_log)
    return [commit for commit in commits if should_include_commit(commit.subject)]


def determine_bump(commits: list[CommitEntry]) -> BumpLevel:
    has_minor = False
    for commit in commits:
        combined = f"{commit.subject}\n{commit.body}".lower()
        if (
            "breaking change" in combined
            or "release:major" in combined
            or "[breaking]" in combined
            or re.match(r"^[a-z]+(?:\([^)]+\))?!:", commit.subject.lower())
        ):
            return "major"
        if (
            "release:minor" in combined
            or commit.subject.lower().startswith("feat:")
            or commit.subject.lower().startswith("feat(")
            or "[feature]" in combined
            or combined.startswith("feature:")
        ):
            has_minor = True
    return "minor" if has_minor else "patch"


def bump_version(version: str, level: BumpLevel) -> str:
    match = SEMVER_RE.match(version)
    if not match:
        raise RuntimeError(f"Unsupported version format: {version}")
    major, minor, patch = (int(part) for part in match.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def update_project_version(pyproject_text: str, version: str) -> str:
    updated, count = PROJECT_VERSION_RE.subn(rf"\g<1>{version}\3", pyproject_text, count=1)
    if count != 1:
        raise RuntimeError("Unable to update project version in pyproject.toml")
    return updated


def extract_core_functions(readme_text: str) -> list[str]:
    match = re.search(r"^## Features\s*$\n(?P<body>.*?)(?:^\#\# |\Z)", readme_text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise RuntimeError("README.md is missing a ## Features section")
    items: list[str] = []
    for line in match.group("body").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:])
    if not items:
        raise RuntimeError("README.md ## Features section does not contain bullet items")
    return items


def render_commit_notes(commits: list[CommitEntry]) -> list[str]:
    if not commits:
        return ["- No user-facing changes were recorded for this release."]
    return [f"- {commit.subject}" for commit in commits]


def render_release_notes(
    version: str,
    previous_tag: str | None,
    commits: list[CommitEntry],
    core_functions: list[str],
) -> str:
    lines = [
        f"# Cyber-Inference {version}",
        "",
        "## Release Notes",
        "",
    ]
    if previous_tag:
        lines.append(f"Changes since `{previous_tag}`:")
    else:
        lines.append("Initial tagged release.")
    lines.extend(["", *render_commit_notes(commits), "", "## Core Functions", ""])
    lines.extend(f"- {item}" for item in core_functions)
    lines.append("")
    return "\n".join(lines)


def render_changelog_entry(
    version: str,
    release_date: str,
    previous_tag: str | None,
    commits: list[CommitEntry],
    core_functions: list[str],
) -> str:
    lines = [f"## [{version}] - {release_date}", "", "### Release Notes", ""]
    if previous_tag:
        lines.append(f"Changes since `{previous_tag}`:")
    else:
        lines.append("Initial tagged release.")
    lines.extend(["", *render_commit_notes(commits), "", "### Core Functions", ""])
    lines.extend(f"- {item}" for item in core_functions)
    lines.append("")
    return "\n".join(lines)


def update_changelog(
    existing_text: str,
    version: str,
    release_date: str,
    previous_tag: str | None,
    commits: list[CommitEntry],
    core_functions: list[str],
) -> str:
    if f"## [{version}] - {release_date}" in existing_text:
        return existing_text

    entry = render_changelog_entry(
        version=version,
        release_date=release_date,
        previous_tag=previous_tag,
        commits=commits,
        core_functions=core_functions,
    ).rstrip()
    trimmed = existing_text.rstrip()
    return f"{trimmed}\n\n{entry}\n"


def write_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def prepare_release(release_notes_file: Path, github_output: Path | None) -> None:
    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    current_version = read_project_version(pyproject_text)
    previous_tag = get_latest_release_tag()
    commits = get_relevant_commits(previous_tag)
    bump = determine_bump(commits)
    version = current_version if previous_tag is None else bump_version(current_version, bump)

    if version != current_version:
        PYPROJECT_PATH.write_text(update_project_version(pyproject_text, version), encoding="utf-8")

    readme_text = README_PATH.read_text(encoding="utf-8")
    core_functions = extract_core_functions(readme_text)
    today = datetime.now(UTC).date().isoformat()

    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")
    updated_changelog = update_changelog(
        existing_text=changelog_text,
        version=version,
        release_date=today,
        previous_tag=previous_tag,
        commits=commits,
        core_functions=core_functions,
    )
    if updated_changelog != changelog_text:
        CHANGELOG_PATH.write_text(updated_changelog, encoding="utf-8")

    release_notes = render_release_notes(
        version=version,
        previous_tag=previous_tag,
        commits=commits,
        core_functions=core_functions,
    )
    release_notes_file.write_text(release_notes, encoding="utf-8")

    output_values = {
        "version": version,
        "tag_name": f"v{version}",
        "bump": bump,
        "previous_tag": previous_tag or "",
        "release_notes_file": str(release_notes_file),
    }
    if github_output is not None:
        write_github_output(github_output, output_values)
    else:
        for key, value in output_values.items():
            print(f"{key}={value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Update release artifacts and emit metadata.")
    prepare_parser.add_argument(
        "--release-notes-file",
        type=Path,
        required=True,
        help="Path for the generated GitHub release notes markdown file.",
    )
    prepare_parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.environ.get("GITHUB_OUTPUT") else None,
        help="Optional GitHub Actions output file path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        prepare_release(
            release_notes_file=args.release_notes_file,
            github_output=args.github_output,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

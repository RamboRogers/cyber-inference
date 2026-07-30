from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_release_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load release script module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release = _load_release_module()


def test_determine_bump_defaults_to_patch():
    commits = [release.CommitEntry(sha="1", subject="Fix model loading timeout", body="")]
    assert release.determine_bump(commits) == "patch"


def test_determine_bump_detects_minor_markers():
    commits = [release.CommitEntry(sha="1", subject="feat: add release dashboard", body="")]
    assert release.determine_bump(commits) == "minor"


def test_determine_bump_detects_major_markers():
    commits = [release.CommitEntry(sha="1", subject="refactor!: break old config path", body="")]
    assert release.determine_bump(commits) == "major"


def test_extract_core_functions_reads_feature_bullets():
    readme = """
## Features

- One
- Two

## Next
"""
    assert release.extract_core_functions(readme) == ["One", "Two"]


def test_repository_readme_context_contract_feeds_release_notes():
    """The fixed context contract must flow from README features into release notes."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    context_feature = (
        "Native model context windows by default, with native, configured, and effective "
        "lengths in `/v1/models`"
    )

    core_functions = release.extract_core_functions(readme)
    notes = release.render_release_notes(
        version="0.2.22",
        previous_tag="v0.2.21",
        commits=[],
        core_functions=core_functions,
    )

    assert context_feature in core_functions
    assert f"- {context_feature}" in notes


def test_update_project_version_changes_first_project_version_only():
    original = '[project]\nname = "cyber-inference"\nversion = "0.2.0"\n'
    updated = release.update_project_version(original, "0.2.1")
    assert 'version = "0.2.1"' in updated
    assert 'version = "0.2.0"' not in updated


def test_update_changelog_prepends_latest_entry():
    existing = "# Changelog\n\nAll notable changes.\n"
    updated = release.update_changelog(
        existing_text=existing,
        version="0.2.1",
        release_date="2026-04-17",
        previous_tag="v0.2.0",
        commits=[release.CommitEntry(sha="1", subject="Fix release plumbing", body="")],
        core_functions=["OpenAI-compatible API"],
    )
    assert "## [0.2.1] - 2026-04-17" in updated
    assert "Fix release plumbing" in updated
    assert "OpenAI-compatible API" in updated

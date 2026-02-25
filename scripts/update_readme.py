#!/usr/bin/env python3
"""
Auto-generate the "Available Skills" table and skill count in README.md.

Reads each skills/*/SKILL.md, extracts frontmatter (name, description, emoji),
and rewrites the section between <!-- SKILLS_TABLE_START --> and
<!-- SKILLS_TABLE_END --> markers in README.md.

Usage:
    python scripts/update_readme.py          # dry-run (prints diff)
    python scripts/update_readme.py --write  # write changes to README.md
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
README = ROOT / "README.md"

SKILLS_TABLE_START = "<!-- SKILLS_TABLE_START -->"
SKILLS_TABLE_END = "<!-- SKILLS_TABLE_END -->"
SKILLS_COUNT_START = "<!-- SKILLS_COUNT_START -->"
SKILLS_COUNT_END = "<!-- SKILLS_COUNT_END -->"


def parse_frontmatter(content: str) -> dict[str, str] | None:
    """Extract YAML-like frontmatter from a SKILL.md file."""
    match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL | re.MULTILINE)
    if not match:
        return None

    frontmatter: dict[str, str] = {}
    for line in match.group(1).strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()
    return frontmatter


def parse_metadata(frontmatter: dict[str, str]) -> dict:
    """Parse the metadata JSON field from frontmatter."""
    raw = frontmatter.get("metadata", "")
    if not raw:
        return {}
    # The frontmatter parser splits on first ":", so metadata value may be truncated.
    # Re-read it from the full frontmatter line.
    return {}


def extract_metadata_from_content(content: str) -> dict:
    """Extract metadata JSON directly from SKILL.md content."""
    match = re.search(r"^metadata:\s*(.+)$", content, re.MULTILINE)
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def get_skill_info(skill_dir: Path) -> dict | None:
    """Extract skill info from a skill directory."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(content)
    if not frontmatter:
        return None

    metadata = extract_metadata_from_content(content)
    openclaw = metadata.get("openclaw", {})

    # Build a short description (first sentence, max 120 chars)
    description = frontmatter.get("description", "")
    short_desc = description.split(". Use this skill")[0]
    if len(short_desc) > 120:
        short_desc = short_desc[:117] + "..."

    # Count scripts
    scripts_dir = skill_dir / "scripts"
    script_count = 0
    if scripts_dir.is_dir():
        script_count = len(list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh")))

    return {
        "name": frontmatter.get("name", skill_dir.name),
        "dir_name": skill_dir.name,
        "description": short_desc,
        "emoji": openclaw.get("emoji", ""),
        "homepage": openclaw.get("homepage", ""),
        "env_vars": openclaw.get("requires", {}).get("env", []),
        "script_count": script_count,
        "user_invocable": frontmatter.get("user-invocable", "false") == "true",
    }


def generate_skills_table(skills: list[dict]) -> str:
    """Generate the markdown table for the Available Skills section."""
    if not skills:
        return "_No skills yet — [add one](CONTRIBUTING.md)!_"

    lines = [
        "| Skill | Description | Env Vars |",
        "|-------|-------------|----------|",
    ]

    for s in sorted(skills, key=lambda x: x["name"]):
        emoji = f'{s["emoji"]} ' if s["emoji"] else ""
        name_link = f'[{emoji}{s["name"]}](skills/{s["dir_name"]}/)'
        env = ", ".join(f"`{v}`" for v in s["env_vars"]) if s["env_vars"] else "—"
        lines.append(f"| {name_link} | {s['description']} | {env} |")

    return "\n".join(lines)


def generate_skills_count(skills: list[dict]) -> str:
    """Generate the skill count badge text."""
    n = len(skills)
    return f"**{n} skill{'s' if n != 1 else ''}** available"


def replace_between_markers(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    """Replace content between two marker comments."""
    pattern = re.escape(start_marker) + r"\n.*?\n" + re.escape(end_marker)
    new_block = f"{start_marker}\n{replacement}\n{end_marker}"
    result, count = re.subn(pattern, new_block, text, flags=re.DOTALL)
    if count == 0:
        print(f"WARNING: Markers {start_marker} / {end_marker} not found in README.md", file=sys.stderr)
    return result


def main() -> int:
    write_mode = "--write" in sys.argv

    if not SKILLS_DIR.is_dir():
        print(f"ERROR: Skills directory not found at {SKILLS_DIR}", file=sys.stderr)
        return 1

    # Collect all skills
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        info = get_skill_info(skill_dir)
        if info:
            skills.append(info)

    print(f"Found {len(skills)} skill(s): {', '.join(s['name'] for s in skills)}")

    # Read current README
    readme_text = README.read_text(encoding="utf-8")

    # Generate replacements
    table = generate_skills_table(skills)
    count = generate_skills_count(skills)

    # Apply replacements
    updated = replace_between_markers(readme_text, SKILLS_TABLE_START, SKILLS_TABLE_END, table)
    updated = replace_between_markers(updated, SKILLS_COUNT_START, SKILLS_COUNT_END, count)

    if updated == readme_text:
        print("README.md is already up to date.")
        return 0

    if write_mode:
        README.write_text(updated, encoding="utf-8")
        print("README.md updated.")
    else:
        print("\nDry run — changes detected. Run with --write to apply.")
        # Show a simple diff summary
        old_lines = readme_text.splitlines()
        new_lines = updated.splitlines()
        for i, (old, new) in enumerate(zip(old_lines, new_lines, strict=False), 1):
            if old != new:
                print(f"  L{i}: - {old}")
                print(f"  L{i}: + {new}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

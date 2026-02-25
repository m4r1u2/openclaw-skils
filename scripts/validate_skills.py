#!/usr/bin/env python3
"""
Validate that all skills in the repository follow the required structure.

Each skill directory under skills/ must contain:
  - SKILL.md  — skill definition with valid frontmatter
  - scripts/  — directory with at least one script

The SKILL.md frontmatter must include:
  - name
  - description
"""

import re
import sys
from pathlib import Path

REQUIRED_FRONTMATTER_KEYS = {"name", "description"}
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def parse_frontmatter(content: str) -> dict[str, str] | None:
    """Extract YAML-like frontmatter from a SKILL.md file."""
    # SKILL.md uses ```skill fenced block with --- delimited frontmatter
    match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL | re.MULTILINE)
    if not match:
        return None

    frontmatter: dict[str, str] = {}
    for line in match.group(1).strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()
    return frontmatter


def validate_skill(skill_dir: Path) -> list[str]:
    """Validate a single skill directory. Returns list of errors."""
    errors: list[str] = []
    skill_name = skill_dir.name

    # Check SKILL.md exists
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        errors.append(f"{skill_name}: Missing SKILL.md")
        return errors

    # Parse and validate frontmatter
    content = skill_md.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(content)

    if frontmatter is None:
        errors.append(f"{skill_name}: SKILL.md has no valid frontmatter (expected --- delimiters)")
    else:
        for key in REQUIRED_FRONTMATTER_KEYS:
            if key not in frontmatter or not frontmatter[key]:
                errors.append(f"{skill_name}: SKILL.md frontmatter missing required key '{key}'")

    # Check scripts directory
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        errors.append(f"{skill_name}: Missing scripts/ directory")
    else:
        scripts = list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh"))
        if not scripts:
            errors.append(f"{skill_name}: scripts/ directory has no .py or .sh files")

    return errors


def main() -> int:
    if not SKILLS_DIR.is_dir():
        print(f"ERROR: Skills directory not found at {SKILLS_DIR}")
        return 1

    skill_dirs = sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir())

    if not skill_dirs:
        print("WARNING: No skill directories found under skills/")
        return 0

    all_errors: list[str] = []
    for skill_dir in skill_dirs:
        errors = validate_skill(skill_dir)
        all_errors.extend(errors)

    # Summary
    print(f"Validated {len(skill_dirs)} skill(s)")
    for skill_dir in skill_dirs:
        status = "PASS" if not any(skill_dir.name in e for e in all_errors) else "FAIL"
        print(f"  {status}  {skill_dir.name}")

    if all_errors:
        print(f"\n{len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("\nAll skills valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

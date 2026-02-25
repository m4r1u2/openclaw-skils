"""Tests for the skill validation script."""

from pathlib import Path

from scripts.validate_skills import parse_frontmatter, validate_skill

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_frontmatter_valid():
    content = """```skill
---
name: test-skill
description: A test skill
---
# Test
```"""
    fm = parse_frontmatter(content)
    assert fm is not None
    assert fm["name"] == "test-skill"
    assert fm["description"] == "A test skill"


def test_parse_frontmatter_missing():
    content = "# Just a heading\nNo frontmatter here."
    fm = parse_frontmatter(content)
    assert fm is None


def test_validate_skill_valid(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('hello')")
    (skill_dir / "SKILL.md").write_text("""```skill
---
name: my-skill
description: A valid skill
---
# My Skill
```""")

    errors = validate_skill(skill_dir)
    assert errors == []


def test_validate_skill_missing_skill_md(tmp_path):
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()

    errors = validate_skill(skill_dir)
    assert any("Missing SKILL.md" in e for e in errors)


def test_validate_skill_missing_scripts(tmp_path):
    skill_dir = tmp_path / "no-scripts"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""```skill
---
name: no-scripts
description: Skill without scripts
---
# No Scripts
```""")

    errors = validate_skill(skill_dir)
    assert any("Missing scripts/" in e for e in errors)

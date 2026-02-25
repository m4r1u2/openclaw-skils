# Contributing to openclaw-skils

Thanks for your interest in contributing! This repository is a community collection of [OpenClaw](https://openclaw.ai) skills.

## Adding a New Skill

Each skill lives in its own directory under `skills/`. Follow this structure:

```
skills/
  your-skill-name/
    SKILL.md          # Skill definition (required)
    scripts/          # Scripts used by the skill (required)
      your_script.py
    requirements.txt  # Python dependencies (optional)
```

### SKILL.md Format

Your `SKILL.md` must start with a fenced frontmatter block:

````markdown
```skill
---
name: your-skill-name
description: What your skill does and when to use it
metadata: { "openclaw": { "emoji": "🔧", "homepage": "https://...", "primaryEnv": "YOUR_API_KEY", "requires": { "env": ["YOUR_API_KEY"] } } }
user-invocable: true
---

# Your Skill Name

Usage instructions, examples, and documentation...
```
````

### Requirements

- **name**: Must match the directory name
- **description**: Clear description of what the skill does and when the AI should use it
- **scripts/**: Must contain at least one `.py` or `.sh` file
- **Python scripts**: Must pass `ruff` linting (see [pyproject.toml](pyproject.toml) for config)

## Development Setup

```bash
# Clone the repo
git clone https://github.com/m4r1u2/openclaw-skils.git
cd openclaw-skils

# Install development dependencies
pip install ruff pytest pyyaml pre-commit

# Set up pre-commit hooks
pre-commit install

# Run linting
ruff check .
ruff format --check .

# Run validation
python scripts/validate_skills.py

# Run tests
pytest tests/ -v
```

## Pull Request Checklist

- [ ] Skill directory follows the required structure
- [ ] `SKILL.md` has valid frontmatter with `name` and `description`
- [ ] Scripts pass `ruff check` and `ruff format --check`
- [ ] `python scripts/validate_skills.py` passes
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Added a `requirements.txt` if your skill needs extra Python packages

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in [pyproject.toml](pyproject.toml). Pre-commit hooks will automatically check your changes.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

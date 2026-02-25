# openclaw-skils

A community collection of [OpenClaw](https://openclaw.ai) skills — ready-to-use AI capabilities you can plug into your OpenClaw agent.

[![CI](https://github.com/m4r1u2/openclaw-skils/actions/workflows/ci.yml/badge.svg)](https://github.com/m4r1u2/openclaw-skils/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What Are OpenClaw Skills?

Each skill is a self-contained package that gives an OpenClaw agent a new capability — from generating AI videos to interacting with external APIs. Skills include a `SKILL.md` definition that tells the agent *when* and *how* to use the skill, plus the scripts that do the actual work.

## Available Skills

<!-- SKILLS_COUNT_START -->
**1 skill** available
<!-- SKILLS_COUNT_END -->

<!-- SKILLS_TABLE_START -->
| Skill | Description | Env Vars |
|-------|-------------|----------|
| [🎬 kling-video](skills/kling-video/) | Generate AI videos using the Kling AI API (klingai.com) | `KLING_ACCESS_KEY`, `KLING_SECRET_KEY` |
<!-- SKILLS_TABLE_END -->

## Quick Start

### Using a Skill

1. Clone this repository (or point your OpenClaw agent to it):
   ```bash
   git clone https://github.com/m4r1u2/openclaw-skils.git
   ```
2. Navigate to the skill you want: `skills/<skill-name>/`
3. Follow the instructions in its `SKILL.md`
4. Install any dependencies listed in `requirements.txt`

### Adding a New Skill

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. In short:

```
skills/
  your-skill-name/
    SKILL.md            # Skill definition with frontmatter
    scripts/            # Your scripts
      your_script.py
    requirements.txt    # Python deps (optional)
```

## Repository Structure

```
├── .github/workflows/   # CI — lint, validate, test
├── scripts/             # Repo-level tooling
│   ├── validate_skills.py
│   └── update_readme.py
├── skills/              # All skills live here
│   └── kling-video/
├── tests/               # Automated tests
├── pyproject.toml       # Ruff linting & project config
├── CONTRIBUTING.md      # How to add a skill
└── LICENSE              # MIT
```

## Development

```bash
pip install ruff pytest pyyaml pre-commit

# Lint
ruff check .
ruff format --check .

# Validate all skills
python scripts/validate_skills.py

# Run tests
pytest tests/ -v

# Set up pre-commit hooks
pre-commit install
```

## License

[MIT](LICENSE)
# PlantCLEF 2026

Research project exploring fine-grained visual categorization in plant species identification using foundation models.

## Quick Start

| Resource | Link |
|----------|------|
| Research concepts | [docs/concepts/](docs/concepts/) |
| Literature references | [docs/references/](docs/references/) |
| Vendor documentation | [docs/vendor/](docs/vendor/) |
| Guides | [docs/guides/](docs/guides/) |

## Repository Structure

```
plantclef-2026/
├── skills/                     # Agent Skills (auto-discovered by compatible agents)
│   ├── pace-cluster/           # PACE Phoenix cluster operations
│   ├── experiment/             # Experiment lifecycle management
│   └── research-docs/          # Research documentation workflow
├── .claude/                    # Claude Code config & slash commands
│   ├── settings.json
│   └── commands/
│       ├── docs/               # /docs:concept, /docs:reference, /docs:vendor
│       ├── experiment/         # /experiment:proposal, /experiment:result, /experiment:validate
│       └── openspec/           # OPSX commands (empty until you run `openspec init`)
├── docs/
│   ├── README.md               # Documentation index
│   ├── guides/                 # How-to documentation
│   ├── _templates/             # Document templates (don't modify)
│   ├── concepts/               # Research ideas
│   ├── references/             # Papers and AI research outputs
│   └── vendor/                 # External tool documentation
├── openspec/
│   ├── config.yaml             # Project context and schema
│   ├── specs/                  # Built specifications
│   └── changes/                # Change proposals
├── user/
│   └── murilo/                 # User workspace
│       └── experiments/
│           └── 000-template/   # Template experiment
│               └── pyproject.toml  # Experiment-level dependencies
├── vendor/                     # Vendored submodules / external repos
├── pyproject.toml              # Workspace root (discovers experiments)
├── AGENTS.md                   # AI assistant guidelines
├── CLAUDE.md                   # Symlink → AGENTS.md
├── ruff.toml                   # Ruff linter/formatter config
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .gitignore
└── LICENSE
```

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Your User Workspace

Each team member has their own workspace under `user/<name>/`:

```bash
cp -r user/example user/<your-name>
```

### 3. Set Up Your Environment

Edit the `pyproject.toml` inside your first experiment (e.g., `user/<your-name>/experiments/001-my-experiment/pyproject.toml`) with your name and dependencies, then:

```bash
# Create virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate

# Install experiment dependencies
uv sync --package <experiment-name>
```

The root `pyproject.toml` defines a uv workspace that auto-discovers all experiments under `user/*/experiments/*`. Each experiment has its own `pyproject.toml` for dependency isolation — different experiments can use different library versions.

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Agent Tooling

This repository uses three systems for AI-assisted workflows. See [Agent Tooling Guide](docs/guides/agent-tooling.md) for full details.

### Skills (Auto-Activated)

[Agent Skills](https://agentskills.io) are auto-discovered by compatible agents (Claude Code, Cursor, Gemini CLI, etc.). Just describe what you need and the agent loads the right skill.

| Skill | Activates when you... |
|-------|----------------------|
| `pace-cluster` | Ask about PACE, Slurm, GPUs, job scripts |
| `experiment` | Want to create, run, or document experiments |
| `research-docs` | Want to capture concepts, papers, or tools |

### Slash Commands (User-Invoked)

Claude Code-specific shortcuts for explicit workflow triggers:

| Command | Purpose |
|---------|---------|
| `/docs:concept` | Create a research concept document |
| `/docs:reference` | Capture knowledge from papers/articles |
| `/docs:vendor` | Document an external tool or dataset |
| `/experiment:proposal` | Propose a new experiment |
| `/experiment:result` | Record experiment results |
| `/experiment:validate` | Validate experiment structure |
| `/opsx:explore` | Investigate ideas before committing to a change |
| `/opsx:new` | Create a new change proposal |
| `/opsx:ff` | Fast-forward all planning artifacts |
| `/opsx:apply` | Implement an approved change |
| `/opsx:archive` | Archive a completed change |

### OpenSpec (Change Management)

Spec-driven development using the [OPSX workflow](https://github.com/Fission-AI/OpenSpec). Run `openspec init` after cloning to generate skills and commands. See [OpenSpec Guide](docs/guides/openspec.md).

## Documentation

See [docs/README.md](docs/README.md) for the full documentation index.

## Contributing

1. Create your user workspace under `user/<your-name>/`
2. Use slash commands to create structured documents
3. Follow the research workflow: Reference → Concept → Experiment → Results
4. See [AGENTS.md](AGENTS.md) for AI assistant conventions

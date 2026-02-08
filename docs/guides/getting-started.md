# Getting Started

How to set up your environment and run your first experiment.

## 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that replaces pip.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

## 2. Create Your User Workspace

Each team member has their own workspace under `user/<name>/`:

```bash
# Copy the example workspace
cp -r user/example user/<your-name>
```

Each experiment within your workspace has its own `pyproject.toml` for dependencies. This gives better isolation — different experiments can use different library versions.

## 3. Set Up Your Environment

```bash
# Create a virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies for a specific experiment
uv sync --package <experiment-name>
```

The root `pyproject.toml` defines a uv workspace that auto-discovers all experiments under `user/*/experiments/*`.

## 4. Initialize OpenSpec (optional)

If you want to use spec-driven change management for architecture decisions:

```bash
npm install -g @fission-ai/openspec@latest
openspec init
```

This generates OPSX commands (e.g., `/opsx:new`, `/opsx:apply`) in `.claude/commands/openspec/`.

## 5. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically on every commit.

## 6. Run Your First Experiment

Use Claude Code to create an experiment proposal:

```
/experiment:proposal
```

This will guide you through:
- Choosing an experiment name and ID
- Writing a hypothesis
- Defining method and success criteria
- Creating a task checklist

## Next Steps

- Read [Claude Commands](claude-commands.md) to learn about all available slash commands
- Read [OpenSpec](openspec.md) to understand spec-driven development
- If using PACE cluster, read [PACE Setup](pace-setup.md)

#!/bin/bash
# Quick environment setup script for RL4VLM
# Usage: source setup_env.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Setting up RL4VLM environment..."

# Activate existing venv or create new one
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "📦 Installing dependencies..."

# Install LLaVA first (has most dependencies)
echo "  → Installing LLaVA..."
pip install -e "./LLaVA" --quiet

# Install other packages (skip pybullet if it fails on macOS)
echo "  → Installing RL packages..."
pip install gym matplotlib h5py accelerate deepspeed --quiet || true

# Try to install pybullet (may fail on macOS, that's okay)
echo "  → Installing pybullet (may fail on macOS)..."
pip install pybullet --quiet || echo "  ⚠️  pybullet installation skipped (not available on macOS)"

# Install stable-baselines3
pip install stable-baselines3 --quiet || true

# Install OpenAI for synthetic data generation
pip install openai --quiet

# Install local packages
echo "  → Installing local packages..."
pip install -e "./VLM_PPO" --quiet || true
pip install -e "./gym-cards" --quiet || true

echo "✅ Environment setup complete!"
echo ""
echo "To activate this environment in the future, run:"
echo "  source $PROJECT_ROOT/.venv/bin/activate"
echo ""
echo "Or add this to your ~/.zshrc:"
echo "  alias rl4vlm='cd \"$PROJECT_ROOT\" && source .venv/bin/activate'"


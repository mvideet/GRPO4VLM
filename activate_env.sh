#!/bin/bash
# Quick activation script - just activates the venv
# Usage: source activate_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"
echo "✅ RL4VLM environment activated!"


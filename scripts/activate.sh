# Source this file (do not execute) to activate the QoQ env with all caches
# redirected into the repo directory. HPC $HOME is full — nothing goes to ~/.
#
#   source scripts/activate.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export REPO_DIR

# shellcheck disable=SC1091
source "$REPO_DIR/miniforge3/etc/profile.d/conda.sh"
conda activate "$REPO_DIR/envs/qoq"

export HF_HOME="$REPO_DIR/hf_cache"
export HF_HUB_CACHE="$REPO_DIR/hf_cache/hub"
export TRANSFORMERS_CACHE="$REPO_DIR/hf_cache"
export HF_DATASETS_CACHE="$REPO_DIR/hf_cache/datasets"
export TORCH_HOME="$REPO_DIR/torch_cache"
export XDG_CACHE_HOME="$REPO_DIR/.cache"
export PIP_CACHE_DIR="$REPO_DIR/pip_cache"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

echo "[activate] env=$CONDA_PREFIX"
echo "[activate] HF_HUB_CACHE=$HF_HUB_CACHE"

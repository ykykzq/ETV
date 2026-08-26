#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${ETV_PYTHON:-python3.12}
VENV=${ETV_TTIR_VENV:-"$ROOT/.venv-ttir"}
UPSTREAM_ROOT=${ETV_UPSTREAM_ROOT:-"$ROOT/build/upstream"}
UV_CACHE=${UV_CACHE_DIR:-"$ROOT/build/uv-cache"}

TRITON_COMMIT=f797708c0626e5f9840ca5b0a98790e2c7cb09ad
NINETOOTHED_COMMIT=efe519d1b12a820e7aa605d775af3d52c8b0d605
NTOPS_COMMIT=9ae4166ad342e4745f0eed13a5a20d069e994fc0

command -v uv >/dev/null 2>&1 || {
    echo "uv is required; install uv >=0.12.0,<0.13" >&2
    exit 2
}
if [ "$(uname -s)" != "Darwin" ]; then
    echo "This source-build helper is for macOS; use the ttir extra on Linux." >&2
    exit 2
fi

mkdir -p "$UPSTREAM_ROOT" "$UV_CACHE"

sync_checkout() {
    name=$1
    url=$2
    commit=$3
    checkout=$4
    mode=$5
    ref=$6

    if [ ! -d "$checkout/.git" ]; then
        if [ "$mode" = "tag" ]; then
            git clone --depth 1 --branch "$ref" "$url" "$checkout"
        else
            git clone "$url" "$checkout"
        fi
    fi
    if [ -n "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]; then
        echo "$name checkout has modified tracked files: $checkout" >&2
        exit 2
    fi
    if [ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]; then
        if [ "$mode" = "tag" ]; then
            git -C "$checkout" fetch --depth 1 origin "refs/tags/$ref:refs/tags/$ref"
        else
            git -C "$checkout" fetch origin
        fi
        git -C "$checkout" checkout --detach "$commit"
    fi
}

env UV_PROJECT_ENVIRONMENT="$VENV" UV_CACHE_DIR="$UV_CACHE" \
    uv sync --locked --inexact --python "$PYTHON_BIN" \
    --extra dev --extra extraction --extra ttir-build

TRITON_ROOT="$UPSTREAM_ROOT/triton-3.7.1"
NINETOOTHED_ROOT="$UPSTREAM_ROOT/ninetoothed"
NTOPS_ROOT="$UPSTREAM_ROOT/ntops"
sync_checkout triton https://github.com/triton-lang/triton.git "$TRITON_COMMIT" "$TRITON_ROOT" tag v3.7.1
sync_checkout ninetoothed https://github.com/InfiniTensor/ninetoothed.git "$NINETOOTHED_COMMIT" "$NINETOOTHED_ROOT" history master
sync_checkout ntops https://github.com/InfiniTensor/ntops.git "$NTOPS_COMMIT" "$NTOPS_ROOT" history master

env UV_CACHE_DIR="$UV_CACHE" TRITON_BUILD_PROTON=OFF \
    TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF MAX_JOBS=4 \
    uv pip install --python "$VENV/bin/python" --no-build-isolation -e "$TRITON_ROOT"
env UV_CACHE_DIR="$UV_CACHE" \
    uv pip install --python "$VENV/bin/python" --no-deps -e "$NINETOOTHED_ROOT"
env UV_CACHE_DIR="$UV_CACHE" \
    uv pip install --python "$VENV/bin/python" --no-deps -e "$NTOPS_ROOT"

"$VENV/bin/python" -c \
    'import torch, triton; from triton._C import libtriton; print(torch.__version__, triton.__version__, libtriton.__file__)'
printf '%s\n' "TTIR environment ready: $VENV"
printf '%s\n' "ntops checkout: $NTOPS_ROOT"

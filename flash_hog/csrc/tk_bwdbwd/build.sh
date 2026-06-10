#!/usr/bin/env bash
# Build libtk_bwdbwd.so — the ThunderKittens double-backward XLA FFI plugin.
#
# Requirements:
#   - CUDA >= 12.8 (nvcc), targeting SM90a (Hopper only)
#   - a ThunderKittens checkout (pinned commit 34b15f7e7012de25ae162c8d9dc85296dd342676):
#       THUNDERKITTENS_PATH=/path/to/ThunderKittens ./build.sh
#   - jax installed in the active python (provides the XLA FFI headers)
#
# Output: libtk_bwdbwd.so next to this script — flash_hog.jax._tk_gpu loads it from
# there by default (override with FLASH_HOG_TK_LIB).
set -euo pipefail
cd "$(dirname "$0")"

TK="${THUNDERKITTENS_PATH:?set THUNDERKITTENS_PATH to a ThunderKittens checkout}"
FFI_INC="$(python -c 'import jax; print(jax.ffi.include_dir())')"

nvcc -shared -Xcompiler -fPIC -std=c++20 -O3 --use_fast_math \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -forward-unknown-to-host-compiler -Xcompiler=-fno-strict-aliasing -Xcompiler=-Wno-psabi \
    -DNDEBUG -DKITTENS_SM90 -gencode arch=compute_90a,code=sm_90a \
    -I"$TK/include" -I"$TK/prototype" -I"$FFI_INC" \
    -L/usr/local/cuda/lib64/stubs -lcuda \
    ffi.cu -o libtk_bwdbwd.so

echo "built $(pwd)/libtk_bwdbwd.so"

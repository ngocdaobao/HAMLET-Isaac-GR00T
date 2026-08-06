#!/bin/bash
# Runtime library path for torchcodec: system FFmpeg 7 + GCC 13 runtime + torch libs.
#
# torchcodec's libtorchcodec_decoder7.so has NO RPATH and links against libavutil.so.59
# & friends, so the dynamic loader can only find them through LD_LIBRARY_PATH. That
# variable is read by ld.so when a process STARTS -- setting it in a subshell, or from
# inside Python, does nothing for the job you launch afterwards. So:
#
#     source torchcodec_setup.sh      # correct: exports land in the current shell
#     bash torchcodec_setup.sh        # WRONG: exports die with the subshell
#
# Anything that starts python (torchrun, srun, sbatch) must be launched from a shell
# that has already sourced this. Dataloader workers inherit it via fork.
#
# Run this file directly (bash torchcodec_setup.sh) to print a verification report.

# ===== GCC runtime =====
export GCC_LIB=/sw/spack/25.04/linux-rocky8-x86_64/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/gcc-13.2.0-gg3swikigb4qfi2xkmpnyo26kb2zwihr/lib64

# ===== FFmpeg =====
export FFMPEG_HOME=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/ffmpeg-7.0.2-i2nq6vl5j6nxrorhdfhu7waxog67oc5o

# ===== PyTorch shared libraries =====
# Prefer the repo venv's interpreter: this may be sourced before the venv is activated.
_TC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_TC_PY="${_TC_ROOT}/.venv/bin/python"
[ -x "$_TC_PY" ] || _TC_PY="$(command -v python3 || command -v python || true)"
# `|| true` so sourcing this from a script running under `set -e` cannot abort the job.
TORCH_LIB="$("$_TC_PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
export TORCH_LIB

# ===== Runtime library search path =====
# Guarded so repeated sourcing does not keep growing the path.
if [ "${TORCHCODEC_ENV_READY:-}" != "1" ]; then
    export PATH="$FFMPEG_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$GCC_LIB:$TORCH_LIB:$FFMPEG_HOME/lib:${LD_LIBRARY_PATH:-}"
    export TORCHCODEC_ENV_READY=1
fi

unset _TC_ROOT _TC_PY

# ===== Verification (only when executed directly, not when sourced) =====
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
    echo "========== Environment =========="
    echo "FFmpeg: $(which ffmpeg)"
    echo "Torch lib: $TORCH_LIB"
    echo "LD_LIBRARY_PATH:"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n'

    echo
    echo "========== Verify =========="
    ffmpeg -version | head -1

    _VERIFY_PY="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"
    [ -x "$_VERIFY_PY" ] || _VERIFY_PY="$(command -v python3 || command -v python)"
    "$_VERIFY_PY" - <<'PY'
import torch
print("Torch:", torch.__version__)
import torchcodec
from torchcodec.decoders import VideoDecoder  # noqa: F401
print("TorchCodec:", torchcodec.__version__, "-- loaded successfully!")
PY

    echo
    echo "NOTE: this was a direct run, so the exports above are gone now."
    echo "      Use 'source torchcodec_setup.sh' before launching training."
fi

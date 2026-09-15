#!/usr/bin/env bash
# Kept only so old launch notes fail with a useful migration message.
set -euo pipefail

echo "Use: python Trace/run_opd_grpo.py train-paired-teacher ..." >&2
echo "The Python entry trains the independent teacher from the shared base checkpoint with upper=real/lower=candidate input." >&2
exit 2

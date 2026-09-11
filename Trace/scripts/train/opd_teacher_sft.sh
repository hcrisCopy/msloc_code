#!/usr/bin/env bash
# Kept only so old launch notes fail safely.  This project no longer trains an
# OPD teacher: use the frozen-SFT paired-video precheck instead.
set -euo pipefail

echo "opd_teacher_sft.sh is retired: the OPD teacher must remain frozen. Running precheck_opd_teacher.sh instead." >&2
exec "$(cd "$(dirname "$0")" && pwd)/precheck_opd_teacher.sh"

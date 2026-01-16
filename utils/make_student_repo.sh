#!/usr/bin/env bash
set -euo pipefail

# 0. Remove all notebooks, except the _student versions (recursively)
find . -name "*.ipynb" ! -name "*_student.ipynb" -print0 | xargs -0 rm -f

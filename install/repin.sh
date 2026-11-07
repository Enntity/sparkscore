#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Move the engine pin in install/atlas-source.json to another Atlas commit.
#
#   install/repin.sh COMMIT [REPOSITORY] [BRANCH]
#
# REPOSITORY defaults to the upstream, https://github.com/Atlas-Inf/atlas.git,
# and BRANCH to main: after the score-path pull request merges, the repin is
#   install/repin.sh <merge commit>
# The script fetches the commit, records its tree, checks that the commit is on
# BRANCH and still carries the score path, and prints what to rebuild and rerun.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
commit=${1:?usage: install/repin.sh COMMIT [REPOSITORY] [BRANCH]}
repo=${2:-https://github.com/Atlas-Inf/atlas.git}
branch=${3:-main}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work"
GIT_LFS_SKIP_SMUDGE=1 git -C "$work" fetch -q --filter=blob:none "$repo" "+refs/heads/$branch:refs/pin/branch" "$commit"
full=$(git -C "$work" rev-parse --verify "$commit^{commit}")
tree=$(git -C "$work" rev-parse "$full^{tree}")
git -C "$work" merge-base --is-ancestor "$full" refs/pin/branch || { echo "$full is not on $repo $branch" >&2; exit 1; }
# The score path SparkScore needs: the /v1/score route and the W4A4 lever.
for path in crates/spark-server/src/api/score.rs crates/spark-model/src/layers/ops/dense_bf16mma_lowp.rs; do
  git -C "$work" cat-file -e "$full:$path" 2>/dev/null || { echo "$full has no $path: not a SparkScore-capable Atlas" >&2; exit 1; }
done
python3 - "$here/atlas-source.json" "$repo" "$branch" "$full" "$tree" <<'EOF'
import json, sys
path, repo, branch, commit, tree = sys.argv[1:]
m = json.load(open(path))
m.update(repository=repo, branch=branch, commit=commit, tree=tree)
open(path, "w").write(json.dumps(m, indent=2) + "\n")
EOF
echo "pinned $repo $branch at $full (tree $tree)"
echo "next: commit install/atlas-source.json, ./start.sh build, ./start.sh serve, then docs/REPRODUCE.md's quick check"

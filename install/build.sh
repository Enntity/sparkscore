#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the SparkScore image: Atlas at the pinned commit (atlas-source.json)
# plus the harness. The tag is derived from the git trees of install/ and
# tools/, so the same checkout always builds the same tag. Never starts a server
# or downloads a model.
#
#   install/build.sh         build the image, then print its tag
#   install/build.sh --tag   print the tag only
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
product=$(cd "$here/.." && pwd)
tree=$( { git -C "$product" rev-parse HEAD:install; git -C "$product" rev-parse HEAD:tools; } | sha1sum | cut -c1-12)
image=${SPARKSCORE_IMAGE_REPO:-sparkscore}:$tree
[[ ${1:-} == --tag ]] && { echo "$image"; exit 0; }
[[ $# == 0 ]] || { echo 'usage: install/build.sh [--tag]' >&2; exit 2; }
[[ -z $(git -C "$product" status --porcelain --untracked-files=all -- install tools) ]] || {
  echo 'install/ or tools/ has local changes; commit them first (the image tag is their git trees)' >&2; exit 2; }
read -r repo commit pin_tree < <(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["repository"], m["commit"], m["tree"])' "$here/atlas-source.json")
[[ $commit =~ ^[0-9a-f]{40}$ && $pin_tree =~ ^[0-9a-f]{40}$ ]] || { echo "install/atlas-source.json: bad pin" >&2; exit 2; }

mkdir -p "${SPARKSCORE_BUILD_DIR:=$HOME/.cache/sparkscore}"
context=$(mktemp -d "$SPARKSCORE_BUILD_DIR/build.XXXXXX")
trap 'rm -rf "$context"' EXIT
# A shallow fetch of the pinned commit, checked against the pinned tree.
git init -q "$context/engine"
GIT_LFS_SKIP_SMUDGE=1 git -C "$context/engine" fetch -q --depth 1 "$repo" "$commit"
GIT_LFS_SKIP_SMUDGE=1 git -C "$context/engine" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
[[ $(git -C "$context/engine" rev-parse 'HEAD^{tree}') == "$pin_tree" ]] || {
  echo "fetched engine is not the pinned Atlas tree $pin_tree" >&2; exit 1; }
git -C "$product" archive HEAD install tools LICENSE NOTICE | tar -x -C "$context"
cp "$here/dockerignore" "$context/.dockerignore"

docker build --build-arg "SOURCE_TREE=$tree" --build-arg "BUILD_JOBS=${ATLAS_BUILD_JOBS:-8}" \
  -f "$context/install/Dockerfile" -t "$image" "$context" >&2
[[ $(docker image inspect -f '{{.Architecture}} {{index .Config.Labels "io.enntity.sparkscore.source-tree"}}' "$image") == "arm64 $tree" ]] || {
  echo "built image $image has the wrong architecture or source-tree label" >&2; exit 1; }
echo "$image"

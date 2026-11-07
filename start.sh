#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SparkScore on one DGX Spark: typed, calibrated decisions from d3-ultra 27B
# NVFP4, served by Atlas.
#
#   ./start.sh                 prepare whatever is missing (image, weights), serve, then smoke-test
#   ./start.sh --precision w4a8   the same, with W4A8 instead of the W4A4 default
#   ./start.sh build           pull the published image, or build it if it isn't published
#   ./start.sh convert         download the published NVFP4 weights (22 GB), checked by checksum
#   ./start.sh stop | status | logs
#   ./start.sh smoke           send one typed request and check the answer
#
# FROM_SOURCE=1 builds the image (Atlas at install/atlas-source.json + the
# harness) and converts d3-ultra yourself (about 96 GB of disk) instead.
#
# Settings (environment or .env): MODEL_ROOT (default ~/models/sparkscore),
# PORT (8015), BIND (127.0.0.1; 0.0.0.0 to serve your network), IMAGE (a
# specific image), SPARKSCORE_API_KEY (bearer key).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
MODEL_ROOT=${MODEL_ROOT:-$HOME/models/sparkscore}
PORT=${PORT:-8015}
BIND=${BIND:-127.0.0.1}
NAME=sparkscore
PRECISION=${PRECISION:-}
if [[ ${1:-} == --precision ]]; then PRECISION=${2:?--precision needs w4a4 or w4a8}; shift 2; fi
[[ -z $PRECISION || $PRECISION == w4a4 || $PRECISION == w4a8 ]] || { echo '--precision must be w4a4 or w4a8' >&2; exit 2; }
say() { printf '\033[1m== %s\033[0m\n' "$*"; }
image() { echo "${IMAGE:-$(install/build.sh --tag)}"; }
model=$MODEL_ROOT/d3-ultra-NVFP4

build() {
  has_image && return
  if [[ ${FROM_SOURCE:-0} != 1 && -z ${IMAGE:-} ]] && docker pull "$(image)"; then return; fi
  say "building $(image) (Atlas from source: 20-40 minutes the first time)"; install/build.sh
}
has_image() { docker image inspect "$(image)" >/dev/null 2>&1; }
convert() {
  [[ -f $model/.converted ]] && return
  if [[ ${FROM_SOURCE:-0} == 1 ]]; then say "converting d3-ultra to NVFP4 in $MODEL_ROOT"; install/convert.sh "$MODEL_ROOT"
  else say "downloading the NVFP4 weights to $MODEL_ROOT"; install/fetch.sh "$MODEL_ROOT"; fi
}
smoke() { python3 install/smoke.py "http://127.0.0.1:$PORT"; }

serve() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  say "serving $(image) on $BIND:$PORT${PRECISION:+ ($PRECISION)}"
  docker run -d --name "$NAME" --gpus all --ipc=host --restart unless-stopped \
    -p "$BIND:$PORT:8015" -v "$model:/model:ro" \
    ${SPARKSCORE_API_KEY:+-e SPARKSCORE_API_KEY} ${PRECISION:+-e SPARKSCORE_PRECISION=$PRECISION} \
    "$(image)" >/dev/null
  say "waiting for the model to load (about a minute)"
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { say "ready: http://$BIND:$PORT/v1/systemone"; return 0; }
    [[ $(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null) == true ]] || { docker logs --tail 40 "$NAME" >&2; echo 'the server exited' >&2; exit 1; }
    sleep 2
  done
  docker logs --tail 40 "$NAME" >&2; echo 'not ready after 20 minutes' >&2; exit 1
}

case ${1:-up} in
  up) build; convert; serve; smoke ;;
  build) build ;;
  convert) convert ;;
  serve) serve ;;
  smoke) smoke ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && say stopped || true ;;
  status) docker ps -a --filter "name=^$NAME\$" --format '{{.Names}}  {{.Status}}  {{.Image}}'
          docker exec "$NAME" python3 -c 'import json; m=json.load(open("/opt/sparkscore/atlas-source.json")); print("engine:", m["repository"], m["branch"], m["commit"])' 2>/dev/null || true
          docker exec "$NAME" sh -c 'echo "precision: ${SPARKSCORE_PRECISION:-w4a4}"' 2>/dev/null || true
          curl -sf "http://127.0.0.1:$PORT/v1/limits" && echo ;;
  logs) docker logs -f --tail 100 "$NAME" ;;
  *) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac

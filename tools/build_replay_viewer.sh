#!/bin/sh
# coworld build hook: produce the static replay viewer bundle in "$1".
# The bundle is prebuilt and checked in at build/static-replay-viewer
# (wasm sim + arena.js adapter, verified against the three recorded
# 50-turn matches), so this just copies it.
set -eu
dest="$1"
mkdir -p "$dest"
cp -R "$(dirname "$0")/../build/static-replay-viewer/." "$dest"/

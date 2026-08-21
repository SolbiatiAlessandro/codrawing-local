#!/bin/sh
# coworld build hook: produce the static replay viewer bundle in "$1".
#
# The bundle IS the live game's viewer. codrawing/game/client/viewer.html
# already satisfies the platform contract - it reads the replay URL from
# ?replay=, fetches those bytes (handling .z/.gz), and renders without a game
# container - and it is a single self-contained file with no external
# references. Shipping it directly means the hosted replay looks exactly like
# the local one and there is no second implementation to drift.
set -eu
dest="$1"
root="$(dirname "$0")/.."
mkdir -p "$dest"
cp "$root/codrawing/game/client/viewer.html" "$dest/index.html"

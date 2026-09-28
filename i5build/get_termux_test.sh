#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# get_termux_test.sh -- the "no curl needed" way in.
#
# If `curl` is missing in Termux, the one-liner in setup_termux_test.sh cannot
# fetch anything, and `pkg install curl` is itself the thing you were trying to
# avoid. This script installs the few tools we need and then runs the real test:
#
#     pkg install -y curl && bash ~/get_termux_test.sh
#
# ...or paste the three lines below straight into the Termux prompt.
#
# ASCII only.
# ============================================================================
set -u

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
HOST="${BNB_HOST:-127.0.0.1}"
PORT="${BNB_PORT:-8899}"
URL="http://$HOST:$PORT/setup_termux_test.sh"
OUT="$HOME/setup_termux_test.sh"

echo "== installing the minimum tooling =="
pkg install -y curl 2>&1 | tail -5

if ! command -v curl >/dev/null 2>&1; then
  echo "FATAL: curl still not available after pkg install."
  echo "       run:  pkg update && pkg install -y curl"
  exit 1
fi

echo
echo "== fetching $URL =="
if ! curl -fsSL --connect-timeout 15 -o "$OUT" "$URL"; then
  echo "FATAL: could not reach the PC at $HOST:$PORT."
  echo "       On the PC, check:  adb reverse tcp:$PORT tcp:$PORT"
  echo "       and that the http server is still serving on port $PORT."
  exit 1
fi
echo "got $(wc -c < "$OUT") bytes"

echo
echo "== running the real test =="
bash "$OUT"

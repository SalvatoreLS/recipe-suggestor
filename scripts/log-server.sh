#!/usr/bin/env bash
# Serve a terminal view over the local network with ttyd.
#
#   ./scripts/log-server.sh log            read-only tail of the log file
#   ./scripts/log-server.sh shell          writable tmux session (full shell access)
#   ./scripts/log-server.sh run -- ./bin/x run a command, tee to the log, serve read-only
#
# The read-only modes (log, run) are served with no login: open the URL and the
# log is there. Set $TTYD_PASS to put HTTP Basic auth in front of them anyway.
# The writable shell mode always requires a password ($TTYD_PASS, min 8 chars).
set -euo pipefail

PORT="${TTYD_PORT:-7681}"
LOG_FILE="${LOG_FILE:-/tmp/recipe-suggestor.log}"
SESSION="${TMUX_SESSION:-work}"
WORKDIR="${WORKDIR:-}"
USER_NAME="${TTYD_USER:-watch}"

usage() {
  cat >&2 <<'USAGE'
usage: log-server.sh <mode> [options] [-- command...]

modes:
  log      serve a read-only live tail of the log file
  shell    serve a writable tmux session (anyone with the password gets a shell)
  run      run the given command, tee its output to the log, serve read-only

options:
  -p PORT    port to listen on (default 7681, or $TTYD_PORT)
  -f FILE    log file (default /tmp/recipe-suggestor.log, or $LOG_FILE)
  -s NAME    tmux session name for shell mode (default work, or $TMUX_SESSION)
  -d DIR     chdir here before running the command (run mode); binaries that
             expect resources/ next to them need -d recipe_suggestor/build
  -h         show this help
USAGE
}

[ $# -ge 1 ] || { usage; exit 2; }
MODE="$1"; shift

while getopts ":p:f:s:d:h" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    f) LOG_FILE="$OPTARG" ;;
    s) SESSION="$OPTARG" ;;
    d) WORKDIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "option -$OPTARG needs a value" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift

command -v ttyd >/dev/null || { echo "ttyd not installed: sudo apt install ttyd" >&2; exit 1; }

# AUTH_ARGS is what gets passed to ttyd: empty means no login prompt at all.
PASS="${TTYD_PASS:-}"
AUTH_ARGS=()
if [ -n "$PASS" ]; then
  AUTH_ARGS=(-c "${USER_NAME}:${PASS}")
fi

lan_ip() {
  local ip
  ip="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+' || true)"
  [ -n "$ip" ] || ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  printf '%s' "$ip"
}

banner() {
  local ip; ip="$(lan_ip || true)"; [ -n "$ip" ] || ip="<this-host-ip>"
  echo
  echo "  url:      http://${ip}:${PORT}"
  if [ -n "$PASS" ]; then
    echo "  user:     ${USER_NAME}"
    echo "  password: ${PASS}"
  else
    echo "  login:    none - the page opens straight onto the log"
  fi
  echo "  mode:     $1"
  echo
  echo "  Anyone on this network can open that URL. Trusted LAN only, never port-forward."
  echo "  Ctrl-C to stop."
  echo
}

case "$MODE" in
  log)
    : >>"$LOG_FILE"
    banner "read-only tail of $LOG_FILE"
    exec ttyd -p "$PORT" -R -O "${AUTH_ARGS[@]}" \
      -t fontSize=16 -t 'theme={"background":"#1a1a1a"}' \
      tail -n 200 -F "$LOG_FILE"
    ;;
  run)
    [ $# -ge 1 ] || { echo "run mode needs a command: log-server.sh run -- ./your_binary" >&2; exit 2; }
    : >>"$LOG_FILE"
    [ -z "$WORKDIR" ] || [ -d "$WORKDIR" ] || { echo "no such directory: $WORKDIR" >&2; exit 2; }
    banner "read-only tail of $LOG_FILE (running: $*${WORKDIR:+ in $WORKDIR})"
    ttyd -p "$PORT" -R -O "${AUTH_ARGS[@]}" \
      -t fontSize=16 -t 'theme={"background":"#1a1a1a"}' \
      tail -n 200 -F "$LOG_FILE" &
    TTYD_PID=$!
    trap 'kill "$TTYD_PID" 2>/dev/null || true' EXIT
    # C++ stdout is fully buffered when it is a pipe, so without stdbuf nothing
    # reaches the log until the program flushes 4KB or exits.
    RUN_CMD=("$@")
    if command -v stdbuf >/dev/null; then RUN_CMD=(stdbuf -oL -eL "$@"); fi
    set +e
    ( [ -n "$WORKDIR" ] && cd "$WORKDIR"; exec "${RUN_CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    set -e
    echo "--- command exited with status $STATUS; server still up, Ctrl-C to stop ---" | tee -a "$LOG_FILE"
    wait "$TTYD_PID"
    ;;
  shell)
    command -v tmux >/dev/null || { echo "tmux not installed: sudo apt install tmux" >&2; exit 1; }
    # A writable shell is remote code execution as this user; no anonymous access.
    if [ ${#PASS} -lt 8 ]; then
      echo "shell mode needs TTYD_PASS set to at least 8 characters" >&2
      exit 1
    fi
    banner "WRITABLE shell, tmux session '$SESSION'"
    exec ttyd -p "$PORT" -O "${AUTH_ARGS[@]}" \
      -t fontSize=16 -t 'theme={"background":"#1a1a1a"}' \
      tmux new -A -s "$SESSION"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; usage; exit 2 ;;
esac

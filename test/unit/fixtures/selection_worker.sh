#!/bin/sh
# A controllable selection worker for the deadline tests.
#
# The supervisor starts a worker as:
#   <worker> --tensor-selection-worker <request> <response>
# and enforces the deadline from its own clock. This fixture decides what the
# worker does from the environment, so a test can reach each terminal outcome
# without waiting for a real solver to be slow, and without a marker inside the
# request the fixture would have to decode.
#
#   FIXTURE_MODE       what this worker does
#   FIXTURE_STALL_S    how long a stalling mode sleeps
#   FIXTURE_REAL       the real worker executable a delegating mode runs
#   FIXTURE_MODES_FILE a file of modes, one per line
#
# With FIXTURE_MODES_FILE set, this worker takes the first line and removes it,
# so a test that runs several selections in one process can give each child its
# own behaviour while the environment stays shared.
#
# Modes:
#   stall_before   sleep without touching the stage file
#   stall_solver   report the solver stage, then sleep
#   stall_replay   report the replay stage, then sleep
#   slow_valid     sleep, then produce a real result if it is still alive
#   crash_partial  write a truncated response and exit nonzero
#   exit_13        produce a complete response, then exit nonzero
#   domain_violation  produce a complete response whose first leaf is outside
#                     the domain the request stated
#   domain_widened    produce a complete response that widens the domain and
#                     puts its leaves inside the widened interval
#   excess_dependence append a dependence record for an input that does not
#                     exist
#   spawn_descendant  start a helper in this process group, then stall, so a
#                     test can check the helper did not outlive the deadline
#   valid          run the real worker to completion
set -u

[ "${1:-}" = "--tensor-selection-worker" ] && shift
request="${1:-}"
response="${2:-}"
stage="${response}.stage"
mode="${FIXTURE_MODE:-valid}"
if [ -n "${FIXTURE_MODES_FILE:-}" ] && [ -s "$FIXTURE_MODES_FILE" ]; then
  # The first line belongs to this child; the rest to the ones after it.
  mode=$(sed -n '1p' "$FIXTURE_MODES_FILE")
  sed -i.bak '1d' "$FIXTURE_MODES_FILE" 2>/dev/null || sed '1d' "$FIXTURE_MODES_FILE" > "$FIXTURE_MODES_FILE.next"
  [ -f "$FIXTURE_MODES_FILE.next" ] && mv "$FIXTURE_MODES_FILE.next" "$FIXTURE_MODES_FILE"
  rm -f "$FIXTURE_MODES_FILE.bak"
fi
stall="${FIXTURE_STALL_S:-30}"

case "$mode" in
  stall_before)
    sleep "$stall"
    ;;
  stall_solver)
    printf 'stage=solver\n' > "$stage"
    sleep "$stall"
    ;;
  stall_replay)
    printf 'stage=replay\n' > "$stage"
    sleep "$stall"
    ;;
  slow_valid)
    sleep "$stall"
    exec "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    ;;
  crash_partial)
    printf 'schema=refractir.tensor-selection.response.v1\nstatus=accepted\n' > "$response"
    exit 3
    ;;
  exit_13)
    "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    exit 13
    ;;
  domain_violation)
    "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    python3 - "$response" <<'PYTHON'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
lines = []
for line in path.read_text().splitlines():
  if line.startswith("param_bytes="):
    # 99 as a little-endian i32, keeping the payload length so only the value
    # disagrees with the domain the request stated.
    payload = line.split("=", 1)[1]
    word = "63000000"
    line = "param_bytes=" + word + payload[len(word) :]
  lines.append(line)
path.write_text("\n".join(lines) + "\n")
PYTHON
    ;;
  domain_widened)
    "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    python3 - "$response" <<'PYTHON'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
lines = []
for line in path.read_text().splitlines():
  if line.startswith("domain="):
    # Claim an interval wide enough for leaves the request never admitted.
    line = "domain=-100,100"
  if line.startswith("param_bytes="):
    payload = line.split("=", 1)[1]
    line = "param_bytes=63000000" + payload[8:]
  lines.append(line)
path.write_text("\n".join(lines) + "\n")
PYTHON
    ;;
  excess_dependence)
    "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    printf 'dependence=%%not_an_input\ndependence_value=1\n' >> "$response"
    ;;
  spawn_descendant)
    sh -c "sleep 5; : > \"${FIXTURE_DESCENDANT:?}\"" &
    sleep "$stall"
    ;;
  valid)
    exec "${FIXTURE_REAL:?FIXTURE_REAL is required}" --tensor-selection-worker "$request" "$response"
    ;;
  *)
    echo "unknown FIXTURE_MODE: $mode" >&2
    exit 2
    ;;
esac
exit 0

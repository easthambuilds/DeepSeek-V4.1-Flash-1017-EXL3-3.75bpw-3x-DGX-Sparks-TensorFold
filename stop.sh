#!/bin/bash
# stop.sh: stop all three ranks and free their GPU memory; each rank's log is gzipped first (rt/logs/archive, newest 10)
. "$(dirname "$0")/scripts/common.sh"
step "Stopping the three ranks"
stop_all
for r in 0 1 2; do running $r && die "rank $r is still running"; done
note "stopped"

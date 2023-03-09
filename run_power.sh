#!/usr/bin/env bash

# This script is expected to be run as root (via sudo), because PowerMetrics
# 	must be run as root.  However, everything else should run using the
#	standard user account (non-privileged).

# The general idea is to run PowerMetrics in the background and capture the
#	process id.  Sleep for a short amount of time and then run the
#	benchmark.  Once the benchmark program finishes, sleep a short
#	amount of time and then kill PowerMetrics to finish up.  Data
#	from PowerMetrics can then be loaded into the analytics system
#	for record keeping and analysis.

# USAGE: sudo ./run_power.sh <fileName> "<commandToRun>"
# Example: sudo ./run_power.sh sleep2seconds.txt "/usr/bin/time sleep 2"

powermetrics -i 50 -o $1.dat --samplers cpu_power &
bg_pid=$!
sleep 1
sudo -u $SUDO_USER $2
#sleep 1

echo "Killing powermetrics background process: $bg_pid"
kill -9 $bg_pid

chown $SUDO_USER $1.dat

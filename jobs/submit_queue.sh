#!/bin/bash
# Submit sbatch files one at a time, waiting for a free slot under the cluster's QOS.
#
# /etc/slurm/job_submit.lua refuses a submission when the user already has
#   - 2 RUNNING jobs on a "large partition" (prod40 and prod80 both count), or
#   - 4 RUNNING jobs overall.
# The check looks at RUNNING jobs only and happens at submit time, so pending jobs are
# free and a job that is already queued may start even if that pushes the count past the
# limit. The usable windows are therefore short: one opens the moment a job completes and
# closes as soon as the scheduler starts the next pending job. This polls the same
# condition the plugin checks and submits the instant it clears.
#
#   nohup setsid ./jobs/submit_queue.sh jobs/a.sbatch jobs/b.sbatch \
#       > logs/submit_queue.log 2>&1 < /dev/null &
#
# Env: SUBMIT_QUEUE_INTERVAL (poll seconds, default 5)

set -uo pipefail
INTERVAL="${SUBMIT_QUEUE_INTERVAL:-5}"
MAX_LARGE=2
MAX_TOTAL=4
LARGE_PARTITIONS="prod40 prod80"

running_counts() {
    # -> "<total running> <running on large partitions>"
    squeue -h -u "$USER" -t RUNNING -o "%P" 2>/dev/null | awk -v large="$LARGE_PARTITIONS" '
        BEGIN { n = split(large, a, " "); for (i = 1; i <= n; i++) is_large[a[i]] = 1 }
        { total++; if ($1 in is_large) big++ }
        END { printf "%d %d\n", total + 0, big + 0 }'
}

for job in "$@"; do
    while :; do
        read -r total big < <(running_counts)
        if (( big < MAX_LARGE && total < MAX_TOTAL )); then
            if out=$(sbatch "$job" 2>&1); then
                echo "$(date '+%F %T')  $job -> $out"
                break
            fi
            echo "$(date '+%F %T')  rejected ($job): ${out##*error: }"
        fi
        sleep "$INTERVAL"
    done
    # Let the scheduler register the new job before evaluating the next one
    sleep "$INTERVAL"
done
echo "$(date '+%F %T')  all jobs submitted"

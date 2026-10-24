#!/bin/bash
# usage: worker.sh <worker_id> <n_workers>
cd "$(dirname "$0")"
export EPOCHS=25 PRE_EPOCHS=10 P_DROP=0.25 OMP_NUM_THREADS=1
i=0
while read -r line; do
  if [ $((i % $2)) -eq $1 ]; then
    python3 run.py $line >> logs/worker$1.log 2>&1
  fi
  i=$((i+1))
done < jobs.txt
echo ALLDONE >> logs/worker$1.log

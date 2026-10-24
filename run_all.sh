#!/bin/bash
# Full OpenSense-HAR pipeline. QUICK=1 runs a 2-epoch smoke test on one fold.
set -e
mkdir -p cache results logs
python3 prep_raw.py cache/raw.pkl
python3 prep_seq.py
python3 prep_depth.py
python3 build_text.py
if [ "${QUICK:-0}" = "1" ]; then
  export EPOCHS=2 PRE_EPOCHS=1
  python3 run.py closed opensense 0 0
  python3 run.py closed xmtf 0 0
  python3 run.py open opensense 0 0 mirror
  python3 tta.py 0 closed
  exit 0
fi
# two parallel workers over the job list (each job is one model, one fold)
bash worker.sh 0 2 & bash worker.sh 1 2 & wait
for f in 0 1 2 3 4; do python3 tta.py $f closed; python3 tta.py $f open; done
python3 analysis.py
python3 bench.py
python3 fig_arch.py; python3 fig_mech.py; python3 figs_results.py

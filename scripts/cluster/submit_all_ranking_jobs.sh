#!/bin/bash
# Submit one ranking-dataset build job per KG, all in parallel.
# Run from the project root on coma:
#   bash scripts/cluster/submit_all_ranking_jobs.sh
#
# To override defaults for a specific KG, edit the loop or invoke
# the sbatch directly:
#   DATASET=fb237_v4 MAX_QUERIES=2000 sbatch scripts/cluster/build_ranking_dataset.sbatch

set -u

DATASETS=(
    aids
    codex_l
    fb237_v4
    hetionet
    lubm
    swdf
    wikidata
    wn18rr_v4
    yago
)

SBATCH_FILE="scripts/cluster/build_ranking_dataset.sbatch"

if [ ! -f "$SBATCH_FILE" ]; then
    echo "ERROR: $SBATCH_FILE not found. Run from project root."
    exit 1
fi

echo "Submitting ranking dataset jobs for ${#DATASETS[@]} KGs..."
for ds in "${DATASETS[@]}"; do
    JOB_OUT="logs/rank_ds_${ds}.out"
    JOB_ERR="logs/rank_ds_${ds}.err"
    mkdir -p logs
    JID=$(DATASET="$ds" sbatch \
        --output="$JOB_OUT" \
        --error="$JOB_ERR" \
        --job-name="rank_ds_${ds}" \
        "$SBATCH_FILE" | awk '{print $NF}')
    echo "  $ds -> job $JID  (logs: $JOB_OUT)"
done

echo ""
echo "Watch with: squeue -u \$USER"
echo "Cancel one: scancel <jobid>"
echo "Cancel all: scancel -u \$USER -n rank_ds_<ds>"

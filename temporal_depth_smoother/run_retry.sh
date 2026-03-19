#!/bin/bash
# Retry wrapper for run_pipeline.py
# Handles transient CUDA/WSL memory failures by re-running with --resume.
# Usage: bash run_pipeline_retry.sh --data-dir /path/to/data [extra args...]

MAX_RETRIES=50
ATTEMPT=0
EXIT_CODE=1

echo "======================================================================="
echo "  PIPELINE RETRY WRAPPER"
echo "  Max attempts: $MAX_RETRIES"
echo "  Args: $@"
echo "======================================================================="

while [ $ATTEMPT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
    ATTEMPT=$((ATTEMPT + 1))

    if [ $ATTEMPT -eq 1 ]; then
        echo -e "\n▶ Attempt $ATTEMPT / $MAX_RETRIES (fresh start)\n"
    else
        echo -e "\n▶ Attempt $ATTEMPT / $MAX_RETRIES (retrying after failure)\n"
        sleep 3  # brief pause to let GPU memory settle after a crash
    fi

    python -m temporal_depth_smoother.run_pipeline \
        --data-dir /mnt/d/DATA/phystwin/temporal_depth_training_data/ \
        --skip-existing \
        --resume \
        "$@"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n✓ Pipeline completed successfully on attempt $ATTEMPT\n"
    else
        echo -e "\n❌ Attempt $ATTEMPT failed (exit code $EXIT_CODE)"
        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo "   Retrying..."
        fi
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "\n======================================================================="
    echo "  ❌ Pipeline failed after $MAX_RETRIES attempts."
    echo "  Check logs above for a non-transient error."
    echo "======================================================================="
    exit 1
fi
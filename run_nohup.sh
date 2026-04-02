#!/bin/bash
# Simpler alternative using nohup - runs independently of SSH

EXPERIMENT_NAME="${1:-experiment_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

echo "Starting training with nohup..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"

# Run with nohup - output goes to log file and nohup.out
nohup bash -c "source venv/bin/activate && python -u latent_mass_flow_gp.py" > "$LOG_FILE" 2>&1 &

PID=$!
echo ""
echo "Process started with PID: $PID"
echo ""
echo "Monitor with:  tail -f $LOG_FILE"
echo "Check status:  ps -p $PID"
echo "Kill with:     kill $PID"
echo ""
echo "Safe to disconnect SSH - training will continue!"

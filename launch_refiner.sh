#!/bin/bash
PROJECT_DIR="/mnt/e/Openfoam/PI-FNO-CFD-Pipeline"
VENV="/mnt/e/Openfoam/venv"
LOG_FILE="$PROJECT_DIR/refiner_training.log"
PID_FILE="$PROJECT_DIR/refiner.pid"

cd "$PROJECT_DIR" || exit 1
source "$VENV/bin/activate" || exit 1

# Kill any existing refiner session
tmux kill-session -t pifno_refiner 2>/dev/null || true

echo "Starting extended PINN Refiner training + evaluation..."
echo "Timestamp: $(date)" > "$LOG_FILE"

# Start refiner inside a persistent detached tmux session (train_ref -> evaluate)
tmux new-session -d -s pifno_refiner "cd $PROJECT_DIR && source $VENV/bin/activate && python -u run_pipeline.py --stage train_ref --epochs 1000 2>&1 | tee -a '$LOG_FILE' && python -u run_pipeline.py --stage evaluate 2>&1 | tee -a '$LOG_FILE'; echo '=== Refiner Run & Evaluation Completed ===' >> '$LOG_FILE'"

TMUX_PID=$(tmux list-panes -t pifno_refiner -F "#{pane_pid}")
echo "$TMUX_PID" > "$PID_FILE"

echo "PINN Refiner launched in persistent tmux session 'pifno_refiner' (PID: $TMUX_PID)."
echo "Log file: $LOG_FILE"
echo "To monitor: tail -f $LOG_FILE"
echo "To attach tmux: tmux attach -t pifno_refiner"

#!/bin/bash
PROJECT_DIR="/mnt/e/Openfoam/PI-FNO-CFD-Pipeline"
VENV="/mnt/e/Openfoam/venv"
LOG_FILE="$PROJECT_DIR/pipeline_run.log"
PID_FILE="$PROJECT_DIR/pipeline.pid"

cd "$PROJECT_DIR" || exit 1
source "$VENV/bin/activate" || exit 1

# Kill any existing pipeline tmux session
tmux kill-session -t pifno_pipeline 2>/dev/null || true

echo "Starting full PI-FNO-CFD Pipeline (Data prep -> SR-FNO -> PINN Refiner -> Evaluation)..."
echo "Timestamp: $(date)" > "$LOG_FILE"

# Start pipeline inside a persistent detached tmux session
tmux new-session -d -s pifno_pipeline "cd $PROJECT_DIR && source $VENV/bin/activate && python -u run_pipeline.py --stage full 2>&1 | tee -a '$LOG_FILE'"

TMUX_PID=$(tmux list-panes -t pifno_pipeline -F "#{pane_pid}")
echo "$TMUX_PID" > "$PID_FILE"

echo "Full pipeline launched in persistent tmux session 'pifno_pipeline' (PID: $TMUX_PID)."
echo "Log file: $LOG_FILE"
echo "To monitor: tail -f $LOG_FILE"
echo "To attach tmux: tmux attach -t pifno_pipeline"

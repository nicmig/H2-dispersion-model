#!/bin/bash
# Run training independently of SSH session with logging and notification

# Configuration
EXPERIMENT_DIR="experiments"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${1:-experiment_$TIMESTAMP}"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"
SUMMARY_FILE="$EXPERIMENT_DIR/${EXPERIMENT_NAME}_summary.txt"
EMAIL="${2:-}"  # Optional: provide email as second argument

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$EXPERIMENT_DIR"

echo "=============================================="
echo "  H2 Dispersion Model - Training Launcher"
echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"
echo "Summary will be saved to: $SUMMARY_FILE"
if [ -n "$EMAIL" ]; then
    echo "Notification email: $EMAIL"
fi
echo "=============================================="
echo ""

# Create the Python training script with summary generation
cat > /tmp/run_training_${TIMESTAMP}.py << 'PYTHON_SCRIPT'
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model')

# Import your training module
import torch
import pandas as pd
from latent_mass_flow_gp import train_h2_dispersion_gp

# Configuration
EXPERIMENT_NAME = "EXPERIMENT_NAME_PLACEHOLDER"
SUMMARY_FILE = Path("SUMMARY_FILE_PLACEHOLDER")
LOG_FILE = Path("LOG_FILE_PLACEHOLDER")

def write_summary(status, details):
    """Write summary file with training results."""
    summary = f"""
================================================================================
H2 DISPERSION MODEL TRAINING - {status}
================================================================================

Experiment Name: {EXPERIMENT_NAME}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {status}

--------------------------------------------------------------------------------
SYSTEM INFO
--------------------------------------------------------------------------------
PyTorch Version: {torch.__version__}
CUDA Available: {torch.cuda.is_available()}
CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}

--------------------------------------------------------------------------------
DETAILS
--------------------------------------------------------------------------------
{details}

--------------------------------------------------------------------------------
LOG LOCATION
--------------------------------------------------------------------------------
Full log: {LOG_FILE}

================================================================================
"""
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(summary)
    print(f"\nSummary written to: {SUMMARY_FILE}")

def main():
    start_time = time.time()
    
    try:
        print("=" * 70)
        print("H2 DISPERSION MODEL TRAINING")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("=" * 70)
        print()
        
        # Load data
        print("Loading data...")
        df = pd.read_csv('data/unified_raw.csv')
        print(f"Loaded {len(df)} rows")
        print(f"Mass flow range: {df['mass_flow'].min():.4f} - {df['mass_flow'].max():.4f}")
        print(f"Time range: {df['time'].min():.1f} - {df['time'].max():.1f}")
        print()
        
        # Training configuration
        config = {
            'n_inducing': 500,
            'n_epochs': 200,
            'learning_rate': 0.01,
            'early_stopping_patience': 20,
            'test_size': 0.15,
            'val_size': 0.15,
        }
        
        print("Training configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
        
        # Train model
        print("Starting training...")
        print("-" * 70)
        
        model, likelihood, history = train_h2_dispersion_gp(
            df,
            n_inducing=config['n_inducing'],
            n_epochs=config['n_epochs'],
            learning_rate=config['learning_rate'],
            early_stopping_patience=config['early_stopping_patience'],
            test_size=config['test_size'],
            val_size=config['val_size'],
        )
        
        elapsed = time.time() - start_time
        
        # Success summary
        details = f"""
Training completed successfully!

Configuration:
  - Inducing points: {config['n_inducing']}
  - Epochs: {config['n_epochs']}
  - Learning rate: {config['learning_rate']}
  - Early stopping patience: {config['early_stopping_patience']}

Training History:
  - Best epoch: {history.get('best_epoch', 'N/A')}
  - Best validation loss: {history.get('best_val_loss', 'N/A'):.6f}
  - Final training loss: {history.get('train_losses', [-1])[-1]:.6f}
  - Training duration: {elapsed/60:.2f} minutes

Model saved to: experiments/model_{EXPERIMENT_NAME}.pt
"""
        
        write_summary("SUCCESS", details)
        
        print()
        print("=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Duration: {elapsed/60:.2f} minutes")
        print(f"Summary: {SUMMARY_FILE}")
        
        return 0
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = traceback.format_exc()
        
        details = f"""
Training FAILED with error:

{str(e)}

Full traceback:
{error_msg}

Duration before failure: {elapsed/60:.2f} minutes
"""
        
        write_summary("FAILED", details)
        
        print()
        print("=" * 70)
        print("TRAINING FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        print(f"Summary: {SUMMARY_FILE}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT

# Replace placeholders with actual values
sed -i "s|EXPERIMENT_NAME_PLACEHOLDER|$EXPERIMENT_NAME|g" /tmp/run_training_${TIMESTAMP}.py
sed -i "s|SUMMARY_FILE_PLACEHOLDER|$SUMMARY_FILE|g" /tmp/run_training_${TIMESTAMP}.py
sed -i "s|LOG_FILE_PLACEHOLDER|$LOG_FILE|g" /tmp/run_training_${TIMESTAMP}.py

# Activate virtual environment and run
(
    source venv/bin/activate
    
    # Run training with unbuffered output
    python -u /tmp/run_training_${TIMESTAMP}.py 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Send email notification if requested
    if [ -n "$EMAIL" ] && [ -f "$SUMMARY_FILE" ]; then
        if command -v mail &> /dev/null; then
            SUBJECT="H2 Model Training $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED') - $EXPERIMENT_NAME"
            mail -s "$SUBJECT" "$EMAIL" < "$SUMMARY_FILE"
            echo "Email notification sent to $EMAIL"
        elif command -v sendmail &> /dev/null; then
            SUBJECT="H2 Model Training $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED') - $EXPERIMENT_NAME"
            {
                echo "To: $EMAIL"
                echo "Subject: $SUBJECT"
                echo "Content-Type: text/plain; charset=UTF-8"
                echo ""
                cat "$SUMMARY_FILE"
            } | sendmail "$EMAIL"
            echo "Email notification sent to $EMAIL"
        else
            echo "Warning: No mail command found. Email notification not sent."
        fi
    fi
    
    # Cleanup temporary script
    rm -f /tmp/run_training_${TIMESTAMP}.py
    
    exit $EXIT_CODE
) &

# Get the background job PID
PID=$!
echo ""
echo "Training started in background with PID: $PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "Summary will be available at:"
echo "  $SUMMARY_FILE"
echo ""
echo "You can now close the SSH connection. The training will continue."

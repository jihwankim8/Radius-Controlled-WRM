set -euo pipefail

SCRIPT="mnist_RC-WRM.py"

OUTDIR="runs/mnist_RC-WRM"

RHO0=0.3
RHO_MAX=0.7
RHO_POINTS=31

SEED=12345
FIXED_GAMMAS=(0.2 0.4 0.6 0.8 1.0)

GAMMA_INIT=0.4
NUM_SEGMENTS=6

EPOCHS=4
BATCH_SIZE=128
LR=1e-3
N_TRAIN=12000
N_CALIB=2000
N_TEST=2000

WRM_STEPS=5
EVAL_WRM_STEPS=12
WRM_STEP_SCALE=1.0
CALIBRATION_BATCHES=8

GAMMA_UPDATE_SAMPLES=2000
GAMMA_UPDATE_BATCHES=16

ETA_GAMMA=0.15
RHO_EMA_BETA=0.5
GAMMA_MIN=0.10
GAMMA_MAX=1.50

GAMMA_ADV_GRID=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.80 1.00 1.25 1.50 2.00)

python "$SCRIPT" --version

python "$SCRIPT" \
  --outdir "$OUTDIR" \
  --rho0 "$RHO0" \
  --rho-max "$RHO_MAX" \
  --rho-points "$RHO_POINTS" \
  --seed "$SEED" \
  --fixed-gammas "${FIXED_GAMMAS[@]}" \
  --gamma-init "$GAMMA_INIT" \
  --num-segments "$NUM_SEGMENTS" \
  --gamma-min "$GAMMA_MIN" \
  --gamma-max "$GAMMA_MAX" \
  --eta-gamma "$ETA_GAMMA" \
  --rho-ema-beta "$RHO_EMA_BETA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --n-train "$N_TRAIN" \
  --n-calib "$N_CALIB" \
  --n-test "$N_TEST" \
  --wrm-steps "$WRM_STEPS" \
  --eval-wrm-steps "$EVAL_WRM_STEPS" \
  --wrm-step-scale "$WRM_STEP_SCALE" \
  --calibration-batches "$CALIBRATION_BATCHES" \
  --gamma-update-samples "$GAMMA_UPDATE_SAMPLES" \
  --gamma-update-batches "$GAMMA_UPDATE_BATCHES" \
  --gamma-adv-grid "${GAMMA_ADV_GRID[@]}" \
  --eval-batches 10

echo ""
echo "Done."
echo "Outputs:"
echo "  $OUTDIR/effective_radius.png"
echo "  $OUTDIR/wasserstein_worstcase_loss.png"
echo "  $OUTDIR/wall_clock.txt"
echo "  $OUTDIR/wall_clock.csv"

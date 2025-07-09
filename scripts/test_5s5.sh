#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=2
# Default Training Parameters
DATA_ROOTPATH="/data/liufy/MPDD-2025-Test/MPDD-Elderly/"
TRAIN_MODEL=(
  "checkpoints/5s_5labels_mfccs+densenet/best_model_5s5a.pth"
  "checkpoints/5s_5labels_mfccs+densenet/best_model_5s5b.pth"
  "checkpoints/5s_5labels_mfccs+densenet/best_model_5s5c.pth"
)
ENSEMBLE_WEIGHTS=(0.6 0.35 0.45)  # Ensemble weights for each model, must sum to 1
AUDIOFEATURE_METHOD="mfccs" # Audio feature type, options {wav2vec, opensmile, mfccs}
VIDEOLFEATURE_METHOD="densenet" # Video feature type, options {openface, resnet, densenet}
SPLITWINDOW="5s" # Window duration, options {"1s", "5s"}
LABELCOUNT=5 # Number of label categories, options {2, 3, 5}
TRACK_OPTION="Track1"
FEATURE_MAX_LEN=5 # Set maximum feature length; pad with zeros if insufficient, truncate if exceeding. For Track1, options {26, 5}; for Track2, options {25, 5}
BATCH_SIZE=1
DEVICE="cuda"
MODE="Testing"

for arg in "$@"; do
  case $arg in
    --data_rootpath=*) DATA_ROOTPATH="${arg#*=}" ;;
    # --train_model=*) TRAIN_MODEL="${arg#*=}" ;;
    --audiofeature_method=*) AUDIOFEATURE_METHOD="${arg#*=}" ;;
    --videofeature_method=*) VIDEOLFEATURE_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    --num_epochs=*) NUM_EPOCHS="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    --mode=*) MODE="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# === 构造模型路径参数 ===
TRAIN_MODEL_ARGS=""
for path in "${TRAIN_MODEL[@]}"; do
  TRAIN_MODEL_ARGS+=" --train_model=$path"
done

ENSEMBLE_WEIGHT_ARGS=""
for w in "${ENSEMBLE_WEIGHTS[@]}"; do
  ENSEMBLE_WEIGHT_ARGS+=" --ensemble_weights=$w"
done

for i in `seq 1 1 1`; do
    cmd="python test_ensemble.py \
        --data_rootpath=$DATA_ROOTPATH \
        $TRAIN_MODEL_ARGS \
        $ENSEMBLE_WEIGHT_ARGS \
        --audiofeature_method=$AUDIOFEATURE_METHOD \
        --videofeature_method=$VIDEOLFEATURE_METHOD \
        --splitwindow_time=$SPLITWINDOW \
        --labelcount=$LABELCOUNT \
        --track_option=$TRACK_OPTION \
        --feature_max_len=$FEATURE_MAX_LEN \
        --batch_size=$BATCH_SIZE \
        --mode=$MODE \
        --device=$DEVICE"

    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done

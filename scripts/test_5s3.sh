#!/usr/bin/env bash
set -e

# Default Training Parameters
DATA_ROOTPATH="/data/liufy/MPDD-2025-Test/MPDD-Elderly/"
TRAIN_MODEL="checkpoints/5s_3labels_mfccs+densenet/best_model_5s3.pth"
AUDIOFEATURE_METHOD="mfccs" # Audio feature type, options {wav2vec, opensmile, mfccs}
VIDEOLFEATURE_METHOD="densenet" # Video feature type, options {openface, resnet, densenet}
SPLITWINDOW="5s" # Window duration, options {"1s", "5s"}
LABELCOUNT=3 # Number of label categories, options {2, 3, 5}
TRACK_OPTION="Track1"
FEATURE_MAX_LEN=5 # Set maximum feature length; pad with zeros if insufficient, truncate if exceeding. For Track1, options {26, 5}; for Track2, options {25, 5}
BATCH_SIZE=1
DEVICE="cuda"
MODE="Testing"
export CUDA_VISIBLE_DEVICES=2
for arg in "$@"; do
  case $arg in
    --data_rootpath=*) DATA_ROOTPATH="${arg#*=}" ;;
    --train_model=*) TRAIN_MODEL="${arg#*=}" ;;
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

for i in `seq 1 1 1`; do
    cmd="python test.py \
        --data_rootpath=$DATA_ROOTPATH \
        --train_model=$TRAIN_MODEL \
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

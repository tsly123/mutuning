#!/bin/bash

# custom config
# DATA=/path/to/datasets
#TRAINER=UPT
#TRAINER=VPT
# TRAINER=CoOp
TRAINER=$1

root=/project/hnguyen2/stly/code/datasets/prompting/medmnist
#root=/project/hnguyen2/stly/code/datasets/prompting/deeplesion
#root=/project/hnguyen2/stly/code/prompting/logs_MVLPT/medmnist_ori
#root=/project/hnguyen2/stly/code/datasets/grand_challenges

output_dir=$9
lr=$6
eval_only=$7
get_feat=${10}

# DATASET=$1 # ['hateful-memes', 'cifar-10', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'resisc45_clip', 'country211', 'food-101', 'stanford-cars', 'fgvc-aircraft-2013b-variants102', 'caltech-101', 'dtd', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'eurosat_clip', 'fer-2013', 'kitti-distance']
CFG=$2  # config file
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (5, 20, 50)

# DATASET="Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
# DATASET="ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
PRETRAIN_DATASET="ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
DATASET=$8
#DATASET="Vesselmnist3d"

#MODEL_PATH='/project/hnguyen2/stly/code/prompting/pretrained'
#MODEL_DIR="--model-dir ${MODEL_PATH}/${PRETRAIN_DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp/"
#MODEL_DIR="--model-dir /project/hnguyen2/stly/code/prompting/pretrained/ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101/UPT/vit_b16_20shots/nctx4_csc_ctp/seed1/"
#MODEL_DIR="--model-dir /project/hnguyen2/stly/code/prompting/pretrained/ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101/CoOp/vit_b16_20shots/nctx16_csc_ctp/seed1/"



MODEL_PATH='/project/hnguyen2/stly/code/prompting/pretrained'
MODEL_DIR="--model-dir ${MODEL_PATH}/${PRETRAIN_DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp/"

# for SEED in 1 2 3
# for SEED in 1
for SEED in $5
do
    DIR=$output_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    if [ $TRAINER = "Baseline_scratch" ]; then
      AVG_DIR=$DIR
    else
      AVG_DIR=$output_dir/${DATASET}/${TRAINER}/avg_ckpt
    fi

    if $get_feat ; then
      feat_cmd="--get-feat ${AVG_DIR}"
    else
      feat_cmd="--get-feat None"
    fi

    if $eval_only ; then
      eval_cmd="--eval-only ${DIR}"
    else
      eval_cmd="--eval-only None"
    fi

    if [ $TRAINER = "UPT" ]; then
        python3 train_mvlpt_clip/train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MVLPT \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        --dataset-coop \
        --lr ${lr} \
        ${eval_cmd} \
        ${feat_cmd} \
        ${MODEL_DIR} \
        TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    elif  [ $TRAINER = "VPT" ]; then
        python3 train_mvlpt_clip/train.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
#############################################################
    elif  [ $TRAINER = "Baseline_clip" ]; then
        python3 train_mvlpt_clip/train_baseline_clip.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_clip_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    elif  [ $TRAINER = "Baseline" ]; then
        python3 train_mvlpt_clip/train_baseline.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    elif  [ $TRAINER = "Baseline_last" ]; then
        python3 train_mvlpt_clip/train_baseline_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    elif  [ $TRAINER = "Baseline_clip_bias" ]; then
        python3 train_mvlpt_clip/train_baseline_clip_bias.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_clip_bias_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    elif  [ $TRAINER = "Baseline_bias" ]; then
        python3 train_mvlpt_clip/train_baseline_bias.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_bias_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_bias" ]; then
        python3 train_mvlpt_clip/train_mvlpt_bias.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_bias_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_ss" ]; then
        python3 train_mvlpt_clip/train_mvlpt_ss.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_ss_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_last" ]; then
        python3 train_mvlpt_clip/train_mvlpt_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_bias_last" ]; then
        python3 train_mvlpt_clip/train_mvlpt_bias_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_bias_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "Baseline_scratch" ]; then
        python3 train_mvlpt_clip/train_baseline_scratch.py \
         --root $root \
         --seed ${SEED} \
         --trainer Baseline_scratch_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_ss_last" ]; then
        python3 train_mvlpt_clip/train_mvlpt_ss_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_ss_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_ss_bias" ]; then
        python3 train_mvlpt_clip/train_mvlpt_ss_bias.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_ss_bias_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_ss_bias_last" ]; then
        python3 train_mvlpt_clip/train_mvlpt_ss_bias_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_ss_bias_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_lora" ]; then
        python3 train_mvlpt_clip/train_mvlpt_lora.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_lora_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "VPT_lora_last" ]; then
        python3 train_mvlpt_clip/train_mvlpt_lora_last.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT_lora_last_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         ${MODEL_DIR} \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "Scalebias" ]; then
        python3 train_mvlpt_clip/train_scalebias.py \
         --root $root \
         --seed ${SEED} \
         --trainer Scalebias_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
  elif  [ $TRAINER = "Lora" ]; then
        python3 train_mvlpt_clip/train_lora.py \
         --root $root \
         --seed ${SEED} \
         --trainer Lora_trainer \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --lr ${lr} \
         ${eval_cmd} \
         ${feat_cmd} \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
#############################################################
  else
      python3 train_mvlpt_clip/train.py \
      --root $root \
      --seed ${SEED} \
      --trainer MVLPT \
      --config-file configs/trainers/MVLPT/${CFG}.yaml \
      --output-dir ${DIR} \
      --dataset ${DATASET} \
      --shots ${SHOTS} \
      --dataset-coop \
      --lr ${lr} \
      ${eval_cmd} \
      ${feat_cmd} \
      ${MODEL_DIR} \
      TRAINER.MVLPT.VPT.N_CTX 0 \
      TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
      TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
      TRAINER.MVLPT.COOP.CSC False \
      TEST.NO_TEST False \
  TEST.FINAL_MODEL "best_val" \
      TRAINER.CUT_CONTEXTLEN True
  fi
done

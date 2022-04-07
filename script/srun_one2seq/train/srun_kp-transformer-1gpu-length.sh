#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=train-length-kp20k-transformer-L4H8-DIM512-LR05-DO02-TTTT-TFB1
#SBATCH --output=slurm_output/train-length-kp20k-transformer-L4H8-DIM512-LR05-DO02-TTTT-TFB1.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore
#module load cuda/10.0.130
#module load gcc/6.3.0
#module load python/anaconda3.6-5.2.0
#source activate py36
#module unload python/anaconda3.6-5.2.0

# Run the job
export DATA_NAME="kp20k"
export TOKEN_NAME="meng17"
export TARGET_TYPE="length"
export MASTER_PORT=10000

export LAYER=4
export HEADS=8
export EMBED=512
export HIDDEN=512
export BatchSize=4096
export ValidBatchSize=64
export TrainSteps=200000
export CheckpointSteps=10000

#export LearningRate="2.0"
export LearningRate="0.5"
export Dropout="0.2"

export Copy=true
export ReuseCopy=true
export Cov=true
export PositionEncoding=true

export ShareEmbeddings=true
export CopyLossBySeqLength=false

export ContextGate="both"
export InputFeed=1

export EXP_NAME="$DATA_NAME-$TOKEN_NAME-$TARGET_TYPE-transformer-BS$BatchSize-LR$LearningRate-Layer$LAYER-Heads$HEADS-Dim$HIDDEN-Emb$EMBED-Dropout$Dropout-Copy$Copy-Reuse$ReuseCopy-Cov$Cov-PE$PositionEncoding-Context$ContextGate-IF$InputFeed"

export PATHON_PATH="/ihome/pbrusilovsky/rum20/.conda/envs/py36/bin/"
export ROOT_PATH="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg"
export DATA_PATH="data/keyphrase/$TOKEN_NAME/$DATA_NAME"
export MODEL_PATH="models/keyphrase/$TOKEN_NAME/$EXP_NAME"
export LOG_PATH="output/keyphrase/$TOKEN_NAME/$EXP_NAME.log"
mkdir -p "output/keyphrase/$TOKEN_NAME/"
export TENSORBOARD_PATH="runs/keyphrase/$TOKEN_NAME/$EXP_NAME/"

cmd="python train.py -config config/train/config-transformer-keyphrase-crc.yml -exp $EXP_NAME -data $DATA_PATH -save_model $MODEL_PATH -log_file $LOG_PATH -tensorboard_log_dir $TENSORBOARD_PATH -tgt_type $TARGET_TYPE -batch_size $BatchSize -valid_batch_size $ValidBatchSize -train_steps $TrainSteps -save_checkpoint_steps $CheckpointSteps -layers $LAYER -heads $HEADS -word_vec_size $EMBED -rnn_size $HIDDEN -learning_rate $LearningRate -dropout $Dropout -context_gate $ContextGate -input_feed $InputFeed -master_port $MASTER_PORT"

if [ "$Copy" = true ] ; then
    cmd+=" -copy_attn"
fi
if [ "$ReuseCopy" = true ] ; then
    cmd+=" -reuse_copy_attn"
fi
if [ "$Cov" = true ] ; then
    cmd+=" -coverage_attn"
fi
if [ "$PositionEncoding" = true ] ; then
    cmd+=" -position_encoding"
fi
if [ "$ShareEmbeddings" = true ] ; then
    cmd+=" -share_embeddings"
fi
if [ "$CopyLossBySeqLength" = true ] ; then
    cmd+=" -copy_loss_by_seqlength"
fi

#cmd+=" > output/keyphrase/$TOKEN_NAME/nohup_$EXP_NAME.log &"

echo $TARGET_TYPE
echo $cmd

$cmd
#$cmd
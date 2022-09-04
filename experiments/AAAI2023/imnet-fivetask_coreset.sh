#DEFAULTGPU=0
#GPUID=${1:-$DEFAULTGPU}
#DEFAULTDR="datasets"
DATAROOT="/data"

# benchmark settings
DATE=Imnet_full
SPLIT=10
OUTDIR=outputs/${DATE}/DFCIL-fivetask/ImageNet-50

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=5

# hard coded inputs
REPEAT=1
SCHEDULE="30 60 80 90 100"
PI=50000
MODELNAME=resnet18
BS=128
WD=0.0001
MOM=0.9
OPT="SGD"
LR=0.1


#########################
#  Coreset  #
#########################


# Naive Rehearsal
python3 -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type default --learner_name NormalNN \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/rehearsal "$@"

# LwF - Coreset
python3 -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1 --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name LWF \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_coreset "$@"

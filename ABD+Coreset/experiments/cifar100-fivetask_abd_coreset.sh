# CIFAR-100 five task shell script
# process inputs
# command : bash experiments/cifar100-fivetask.sh --gpuid $GPUID
DEFAULTGPU=0
GPUID=4

# benchmark settings
DATE=IJCAI2023
SPLIT=20
OUTDIR=outputs/${DATE}/DFCIL-fivetask/CIFAR100

###############################################################

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1

# hard coded inputs
REPEAT=3
SCHEDULE="100 150 200 250"
PI=10000
MODELNAME=resnet32
BS=128
WD=0.0002
MOM=0.9
OPT="SGD"
LR=0.1
 
python3 -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1e-1 --memory 50000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name ABD_Coreset \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/ABD_Coreset
'''
# LwF - Coreset
python -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1 --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name LWF \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_coreset

# E2E
python -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1 --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name ETE --DW \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/e2e

# BiC
python -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1 --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name BIC --DW \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/bic

'''

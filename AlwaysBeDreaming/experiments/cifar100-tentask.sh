# sh experiments/cifar100-tentask.sh n

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}

# benchmark settings
#DATE=ICCV2021
#DATE=ResNet34
DATE=HardKD_ResNet32_100SP_real_fake_loss3_notdivide_ysize
SPLIT=10
OUTDIR=outputs/${DATE}/DFCIL-tentask/CIFAR100

###############################################################

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="100 150 200 250"
PI=10000
MODELNAME=resnet32
#MODELNAME=ResNet34
BS=128
WD=0.0002
MOM=0.9
OPT="SGD"
LR=0.1
 
#########################
#         OURS          #
#########################

# Full Method
CUDA_VISIBLE_DEVICES=4 python3 -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1e-1 --memory 0 --model_name $MODELNAME --model_type resnet \
    --learner_type datafree --learner_name AlwaysBeDreaming \
    --gen_model_name CIFAR_GEN --gen_model_type generator \
    --beta 1 --power_iters $PI --deep_inv_params 1e-3 5e1 1e-3 1e3 1 \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/abd



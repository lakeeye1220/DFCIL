# sh experiments/tinyimnet-twentytask.sh n path/to/tiny_imgnet

# process inputs
DEFAULTGPU=0
GPUID=0
DATAROOT="../data/dataset"

# benchmark settings
DATE=AAAI2023
SPLIT=5
OUTDIR=outputs/${DATE}-test/DFCIL-twentytask/TinyImageNet100

###############################################################

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1

# hard coded inputs
REPEAT=1
SCHEDULE="1"
PI=1
MODELNAME=resnet32
BS=128
WD=0.0002
MOM=0.9
OPT="SGD"
LR=0.1
 
# #########################
# #         OURS          #
# #########################

python -u run_dfcil.py --dataset TinyImageNet100 --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 5e-2 --memory 0 --model_name $MODELNAME --model_type resnet \
    --learner_type datafree --learner_name ISCF \
    --gen_model_name TINYIMNET_GEN --gen_model_type generator \
    --beta 1 --power_iters $PI --deep_inv_params 1e-3 5e1 1e-3 1e3 1 \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/iscf \
    --sp_mu 100 --weq_mu 1
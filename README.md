20 Task
```
    python3 -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid 1 --repeat 1 \
    --first_split_size 5 --other_split_size 5 --schedule 100 150 200 250 --schedule_type decay --batch_size 128 \
    --optimizer SGD --lr 0.1 --momentum 0.9 --weight_decay 0.0002 \
    --mu 1e-1 --memory 0 --model_name resnet32 --model_type resnet \
    --learner_type datafree --learner_name AlwaysBeDreamingBalancing \
    --gen_model_name CIFAR_GEN --gen_model_type generator \
    --beta 1 --power_iters 10000 --deep_inv_params 1e-3 5e1 1e-3 1e3 1 \
    --overwrite 0 --max_task -1 --log_dir outputs/balancing_mu1_l2_abdkd0_middle_kd1_mu100/DFCIL-twentytask/CIFAR100/abd \
    --balancing --balancing_mu 1 --middle --balancing_loss_type l2 
```
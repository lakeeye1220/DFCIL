```
python -u run_dfcil.py --dataset CIFAR100 --train_aug --rand_split --gpuid 0 --repeat 1     --first_split_size 20 --other_split_size 20 --schedule 100 150 200 250 --schedule_type decay --batch_size 128     --optimizer SGD 
--lr 0.1 --momentum 0.9 --weight_decay 0.0002     --mu 1e-1 --memory 0 --model_name resnet32 --model_type resnet     --learner_type datafree --learner_name AlwaysBeDreamingBalancing     --gen_model_name CIFAR_GEN --gen_model_type generator     --beta 1 --power_iters 10000 --deep_inv_params 1e-3 5e1 1e-3 1e3 1     --overwrite 0 --max_task -1 --middle_kd_type sp --middle_index real_fake --kd_type hkd_yj --middle_mu 33.3 --kd_index real_fake --ft 
--balancing
```
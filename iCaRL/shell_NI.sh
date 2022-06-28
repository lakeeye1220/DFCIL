#teacher="resnet34"
#student="resnet50"

#CUDA_VISIBLE_DEVICES=5 python3 inversion_main.py --numclass 20 --task_size 20 --epochs 200 --lr 2.0 --seed 10 --prefix fix_ical_
#CUDA_VISIBLE_DEVICES=3 python3 inversion_main.py --numclass 10 --task_size 10 --epochs 200 --lr 1.0 --seed 10 --prefix fix_ical_
CUDA_VISIBLE_DEVICES=4 python3 inversion_main.py --numclass 5 --task_size 5 --epochs 200 --lr 1.0 --seed 10 --prefix fix_ical

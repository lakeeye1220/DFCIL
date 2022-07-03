# RelatedWork

- How to Implement (iCaRL)
```
    python main.py train --train_mode icarl --gpu_ids 0,1 --model resnet32
```

- How to Implement (EEIL)
```
    python main.py train --train_mode eeil --gpu_ids 2 --task_size 10 --model resnet32 --batch_size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr_steps 100,150,200 --weight_decay 0.0002
```

- How to Implement (EEIL+NI)
```
    python main.py train --train_mode eeil --gpu_ids 1 --task_size 10 --model resnet34 --batch_size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr_steps 100,150,200 --weight_decay 0.0002 --natural_inversion True --inversion_epochs 2000
``` 
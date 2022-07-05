# RelatedWork

- How to Implement (iCaRL)
```
    python main.py train --train-mode icarl --gpu-ids 0,1 --model resnet32
```

- How to Implement (EEIL)
```
    python main.py train --train-mode eeil --gpu-ids 2 --task-size 10 --model resnet32 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002
```

- How to Implement (EEIL+NI)
```
    python main.py train --train-mode eeil --gpu-ids 1 --task-size 10 --model resnet34 --batch-size 128 --lr 0.1 --gamma 0.1 --epochs 250 --lr-steps 100,150,200 --weight-decay 0.0002 --natural-inversion True --inversion_epochs 2000
``` 
# TSR
#### Train:

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/pbvs.yml --scale 2 (For Pre-trained)
>
> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/pbvs.yml  --scale 4 --pre_trained model_000120.pth



#### Test

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/pbvs.yml  --scale 4 --test_only


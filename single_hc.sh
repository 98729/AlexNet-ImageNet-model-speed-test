#!/bin/bash

source env_hc.sh $OMPI_COMM_WORLD_LOCAL_RANK
# echo $NCCL_SOCKET_IFNAME

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# APP="python3.8 -u train.py --img 640 --batch 300 --cfg /public/home/aicao/zhubin/non-deep-yolov5/models/yolov5s_2bone2.yaml  --data /public/home/aicao/zhubin/non-deep-yolov5/data/coco128.yaml --weights  /public/home/aicao/zhubin/non-deep-yolov5/yolov5s.pt  --batch-size 64 --dist_url tcp://${1}:43168 --world_size=${comm_size} --rank=${comm_rank}"
APP="python3 -u -m torch.distributed.launch model_batch64.py --master_addr=${MASTER_ADDR}  --master_port=${MASTER_PORT}  --batch-size 64 --dist_url tcp://${1}:29500 --world_size=${comm_size} --local_rank=${comm_rank}"

#/public/home/actqrzwa6p/lizongshu/yolov5/runs/train/exp71/weights


#APP="python -u train.py --data coco.yaml --cfg yolov5x.yaml --weights '' --batch-size 16 --dist_url tcp://${1}:43168 --world_size=${comm_size} --rank=${comm_rank}"

##python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
#   echo numactl --cpunodebind=0 --membind=0 ${APP}
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
#   echo numactl --cpunodebind=1 --membind=1 ${APP}
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
#   echo numactl --cpunodebind=2 --membind=2 ${APP}
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
#   echo numactl --cpunodebind=3 --membind=3 ${APP}
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac
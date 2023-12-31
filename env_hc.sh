module rm compiler/rocm/2.9
module unload compiler/rocm/2.9
module load apps/PyTorch/1.7-dynamic/hpcx-2.7.4-gcc-7.3.1-rocm3.9
#module load  apps/PyTorch/1.6.0a0/hpcx-2.4.1-gcc-7.3.1-rocm3.3
#module load /public/software/modules/compiler/rocm/3.3
#module load apps/anaconda3/2019.10
module load compiler/rocm/3.9.1
#source activate /public/home/aicao/anaconda3/envs/lzs/

#export NCCL_SOCKET_IFNAME=ib$1
#export NCCL_SOCKET_IFNAME=ib1
export NCCL_SOCKET_IFNAME=eno1

export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

export HSA_USERPTR_FOR_PAGED_MEM=0
# export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
# export MIOPEN_SYSTEM_DB_PATH=/temp/pytorch-miopen-2.8
export MIOPEN_SYSTEM_DB_PATH=/public/software/compiler/rocm/rocm-3.9.1/miopen/share/miopen/db/
#export MIOPEN_SYSTEM_DB_PATH=/public/home/aicao/lsder/new_roma/market/module/miopen
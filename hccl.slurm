#!/bin/bash
#SBATCH -J cui
#SBATCH -p normal
#SBATCH -N 64
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -o out.%j
#SBATCH -e err.%j

source env_hc.sh
#module rm compiler/rocm/2.9
#module unload compiler/rocm/2.9
#module load compiler/rocm/3.9.1
module load apps/PyTorch/1.7-dynamic/hpcx-2.7.4-gcc-7.3.1-rocm3.9
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/lib:$LD_LIBRARY_PATH
# echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n"

# declare -a hostnames=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
# PS_HOSTS=${hostnames[0]}:6001
# PS_HOSTS=${hostnames[3]}:6001
# WORKER_HOSTS=""
# for host in ${hostnames[@]:0:3}
# do
#       WORKER_HOSTS=$WORKER_HOSTS$host":6001,"
# done
# let len=${#WORKER_HOSTS}-1
# WORKER_HOSTS=${WORKER_HOSTS:0:len}
# echo $hostnames
# echo $PS_HOSTS
#######DATA_DIR=./mnist

hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile-dl -f

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)

np=$(($np*4))

nodename=$(cat $hostfile |sed -n "1p")
# echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`

# i=0
# for host in ${hostnames[@]:0:3}
# do
#       echo srun -n 16 -w ${hostnames[$i]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name="worker" --task_index=$i
#       srun -n 16 -w ${hostnames[$i]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS  --job_name="worker" --task_index=$i &
#       i=$[$i+1]
# done

# echo srun -n 16 -w ${hostnames[3]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS  --worker_hosts=$WORKER_HOSTS --job_name="ps" --task_index=0
# srun -n 16 -w ${hostnames[3]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS  --worker_hosts=$WORKER_HOSTS  --job_name="ps" --task_index=0
# echo srun -n 1 -w ${hostnames[0]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS --job_name="ps" --task_index=0
# srun -n 1 -w ${hostnames[0]} python3 -m torch.distributed.launch model_batch64.py --ps_hosts=$PS_HOSTS --job_name="ps" --task_index=0

# echo mpirun -np $np --hostfile hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/single_hc.sh $dist_url
mpirun -np $np --hostfile hostfile-dl-$SLURM_JOB_ID  --bind-to none `pwd`/single_hc.sh $dist_url
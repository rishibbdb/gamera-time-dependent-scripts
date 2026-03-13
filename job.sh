#! /bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # match num_threads=32
#SBATCH --mem-per-cpu=4000mb      # 32 * 4000mb = 128GB total
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --job-name=gamera
#SBATCH -o /lustre/hawcz01/scratch/userspace/rbabu/astroimage/gamera-time-dependent-scripts/test-hessfit.txt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /data/disk01/home/rbabu/hawc_software/miniconda3/etc/profile.d/conda.sh
conda activate gamera

cd /lustre/hawcz01/scratch/userspace/rbabu/astroimage/gamera-time-dependent-scripts

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

python beta-test-small.py | tee /lustre/hawcz01/scratch/userspace/rbabu/astroimage/gamera-time-dependent-scripts/pylog-hessfit.txt
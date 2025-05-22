#!/bin/bash
#SBATCH --qos=boost_qos_dbg                    # QoS Max Priority, Max 30 minute con questa priority
#SBATCH -p boost_usr_prod                      # Specify the partition with GPU
#SBATCH --job-name=LLama3.1-70B-Instruct       # Specify Name of the Job
#SBATCH --time=00:30:00                        # Maximum runtime (format HH:MM:SS)
#SBATCH --nodes=4                              # Request 1 node (Il resto delle config dipende da questo)
#SBATCH --ntasks-per-node=1                    # 1 task (process) per node
#SBATCH --gres=gpu:4                           # Request 1 GPU per node (Un nodo ha max 4 GPU)
#SBATCH --cpus-per-task=32                     # Core per task, se i task sono 4 allora si fa tutto diviso 4
#SBATCH --output=%x_output_%j.out              # Output file for job logs (%j is the job ID)
#SBATCH --mail-type=END,FAIL                   # Send mail when job ends or fails
#SBATCH --mail-user=                           # Email for notifications

# Load necessary modules
module load anaconda3
eval "$(conda shell.bash hook)"

# Activate your virtual environment
conda activate SentimentToolKit

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500  # Use a commonly available port

export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Debugging information (optional)
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Run the script with accelerate
accelerate launch \
    --num_processes $WORLD_SIZE \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    main.py

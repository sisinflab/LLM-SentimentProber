#!/bin/bash
#SBATCH --qos=boost_qos_dbg                   # QoS Max Priority, Max 30 minute con questa priority
#SBATCH -p boost_usr_prod                     # Specify the partition with GPU
#SBATCH --job-name=LLama3.1-8B-Instruct       # Specify Name of the Job
#SBATCH --time=00:30:00                       # Maximum runtime (format HH:MM:SS)
#SBATCH --nodes=1                             # Request 1 node (Il resto delle config dipende da questo)
#SBATCH --ntasks-per-node=1                   # 1 task (process) per node
#SBATCH --gres=gpu:4                          # Request 1 GPU per node (Un nodo ha max 4 GPU)
#SBATCH --cpus-per-task=32                    # Core per task, se i task sono 4 allora si fa tutto diviso 4
#SBATCH --output=%x_output_%j.out             # Output file for job logs (%j is the job ID)
#SBATCH --mail-type=END,FAIL                  # Send mail when job ends or fails
#SBATCH --mail-user=                          # Email for notifications

# Load necessary modules
module load anaconda3

# Activate your virtual environment
source activate SentimentToolKit  # Use 'source' for script compatibility

# Run the script with accelerate
python main.py
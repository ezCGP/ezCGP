srun -N 1 -c 8 --mem 128 --gres=gpu:1  -C QuadroRTX4000 --pty bash -i
module load cuda/10.1

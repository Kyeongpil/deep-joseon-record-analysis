NUM_GPU=$1

mpirun -n $NUM_GPU python train.py \
    --hk_batch_size 21 \
    --h_batch_size 36

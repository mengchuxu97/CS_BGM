# BASE_SCRIPT   [
#     "python -u ./src/compressed_sensing.py \\",
#     "    --pretrained-model-dir=./mnist_vae/models/mnist-vae/ \\",
#     "    \\",
#     "    --dataset mnist \\",
#     "    --input-type full-input \\",
#     "    --num-input-images 300 \\",
#     "    --batch-size 50 \\",
#     "    \\",
#     "    --measurement-type gaussian \\",
#     "    --noise-std 0.1 \\",
#     "    --num-measurements 10 \\",
#     "    \\",
#     "    --model-types vae \\",
#     "    --mloss1_weight 0.0 \\",
#     "    --mloss2_weight 1.0 \\",
#     "    --zprior_weight 0.1 \\",
#     "    --dloss1_weight 0.0 \\",
#     "    --lmbd 0.1 \\",
#     "    \\",
#     "    --optimizer-type adam \\",
#     "    --learning-rate 0.01 \\",
#     "    --momentum 0.9 \\",
#     "    --max-update-iter 1000 \\",
#     "    --num-random-restarts 10 \\",
#     "    \\",
#     "    --save-images \\",
#     "    --save-stats \\",
#     "    --print-stats \\",
#     "    --checkpoint-iter 1 \\",
#     "    --image-matrix 0",
#     "",
# ]

cd src

# Lasso

# VAE and VAE+Reg
python create_scripts.py \
    --dataset mnist \
    --pretrained-model-dir ./mnist_vae/models/mnist-vae/ \
    --num-input-images 1024 \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 10 25 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0.0 0.1 \
    --max-update-iter 1000 \
    --num-random-restarts 10 \
    --scripts-base-dir ../mnistscripts_vae

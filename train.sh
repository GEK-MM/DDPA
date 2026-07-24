export CUDA_VISIBLE_DEVICES=1
#${1}
python train.py --dataset rrsisd --ngpu 2  --time 17 --savename Temp --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
#python train.py --dataset refsegrs --ngpu 2  --time 17 --savename Temp --visulize 0 --batch_size 4 --nb_epoch 55 --lr 5e-5
#refsegrs最低学习率为1e-6,其余为1e-7
#python train.py --dataset risbench --ngpu 2  --time 17 --savename Temp --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
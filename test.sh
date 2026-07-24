#ngpu=$1
#CUDA_VISIBLE_DEVICES=${ngpu}
export CUDA_VISIBLE_DEVICES=1
#python test.py --dataset rrsisd --pretrain ./saved_models/f_rrsisd_best.pth --visulize 0
# python test.py --dataset refsegrs --pretrain ./saved_models/refsegrs_best.pth --visulize 0
#ngpu=$1
python test.py --dataset risbench --pretrain ./saved_models/f_risbench_best.pth --visulize 0
#

#ngpu=$1
#CUDA_VISIBLE_DEVICES=${ngpu} python test.py --dataset rrsisd --resume ./saved_models/Temp_model_best.pth.tar --visulize 1
#









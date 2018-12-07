#CUDA_VISIBILE_DEVICES=1 python main.py --lr 1e-4 --model_path baseline4000 --batch_size 64 --print_every 10000 --labeled 50000

#CUDA_VISIBILE_DEVICES=1 python main.py --model_path baseline4000 --print_every 50 --labeled 4000
#CUDA_VISIBILE_DEVICES=0 python main.py --model_path meanteacher4000 --print_every 50 --semi_supervised --mean_teacher --labeled 4000 --consistency 1000.0
CUDA_VISIBILE_DEVICES=0 python main.py --lr 3e-3 --model_path meanteacher4000 --print_every 100 --semi_supervised --mean_teacher --labeled 4000

#CUDA_VISIBILE_DEVICES=1 python main.py --lr 1e-4 --model_path baseline4000 --batch_size 64 --print_every 10000 --labeled 4000 --semi_supervised


#python main.py --lr 1e-5 --batch_size 32 --model_path baseline5000_2
#python main.py --lr 1e-6
#CUDA_VISIBILE_DEVICES=1 python main.py --semi_supervised

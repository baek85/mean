import argparse


parser = argparse.ArgumentParser(description='PyTorch HDR reconstruction Example')

# Hardware
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed to use. Default=123')
parser.add_argument('--num_workers', type= int, default=4,
                    help = 'thue number of workers')                  
# Data
parser.add_argument('--dir_data', type=str, default='../../../../dataset',
                    help='dataset directory')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--semi_supervised', action='store_true',
                    help='semi-supervised learning')
parser.add_argument('--labeled', type=int, default=4000,
                    help='keeped labels')                   
parser.add_argument('--labeled_batch_size', type=int, default=32,
                    help='labeled batchsize')  
# Model
parser.add_argument('--pretrain', type=str, default= '', 
                    help='pretrained model path')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')


# Training
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--mean_teacher', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')  
parser.add_argument('--sntg', action='store_true',
                    help='set this option to training SNTG')   

# Optimization
parser.add_argument('--optim', type=str, default= 'adam', 
                    help='pretrained model path')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='weight decay for SGD optimizer')
parser.add_argument('--alpha', type=float, default=0.99,
                    help='alpha - mean teacher')
parser.add_argument('--consistency', type=float, default=100.0,
                    help='alpha - mean teacher')
parser.add_argument('--consistency_thr', type=int, default=25,
                    help='ramp up length')
parser.add_argument('--lr_decay', type=int, default=100,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
                    
# Loss
parser.add_argument('--loss_type', type=str, default='L1', help='loss type   L1 or GAN')

# Log
parser.add_argument('--model_path', type=str, default="baseline", help='model output path')
parser.add_argument('--print_every', type=int, default=2500,
                    help='how many batches to wait before logging training status')


args = parser.parse_args()




if args.epochs == 0:
    args.epochs = 1e8


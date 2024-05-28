import argparse
import ast

parser = argparse.ArgumentParser()

# ----------------------------------------------------------------------------------------------------
# Data settings
# ----------------------------------------------------------------------------------------------------
parser.add_argument('--data_path', type=str,
                    default="/Users/person/codeScope/BEHI/600A/group project/dataset/DNA_sequence_function_prediction",
                    help='path to save checkpoint')
parser.add_argument('--data_type', default="mini", choices=["mini", "whole"],
                    help="use mini dataset or the whole dataset")
parser.add_argument('--mini_dataset_size', type=int, default=200,
                    help='the size for mini train dataset and valid dataset')
# ----------------------------------------------------------------------------------------------------
# IO settings
# ----------------------------------------------------------------------------------------------------
parser.add_argument('--output', type=str, default="/Users/person/codeScope/BEHI/600A/group project/output",
                    help='path to save checkpoint')
parser.add_argument('--resume', type=str, default=None, help='model path to resume')
parser.add_argument('--save_freq', type=int, default=10, help='save checkpoint each {save_freq} epochs')
parser.add_argument('--save_threshold', type=int, default=80, help='Acc threshold for save model')
parser.add_argument('--auto_resume', type=bool, default=False,
                    help='If true, we will auto check whether checkpoint file in output path and load it!')

# ----------------------------------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='lr')
parser.add_argument('--num_epochs', type=int, default=60, help='epoch')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers used during the training.')
parser.add_argument('--print_freq', type=int, default=10, help='print metric freq')
parser.add_argument('--device', type=int, default=0, help='GPU index used to train')

# ------------------------------------Early stop--------------------------------------------------------
parser.add_argument('--patience', type=int, default=10.,
                    help='if metric not be better for patience epochs, then early stop!')

# ------------------------------------Trick-------------------------------------------------------------
parser.add_argument('--clip_grad', type=float, default=0,
                    help='if the norm of grad is larger than clip_grad, set gard to clip_grad, '
                         'if set to 0, we do not use grad clip ')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='use label smoothing when calculate loss, if set to 0, we do not use label smoothing')

# ------------------------------------Optimizer---------------------------------------------------------
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'rmsprop'],
                    help='optimizer used to update model param')
parser.add_argument('--optimizer_eps', type=float, default=1e-8,
                    help='term added to the denominator to improve numerical stability (default: 1e-8) '
                         '[only used in adamw optimizer]')
parser.add_argument('--optimizer_betas', default=(0.9, 0.999),
                    help='betas (Tuple[float, float], optional): coefficients used for computing running averages '
                         'of gradient and its square (default: (0.9, 0.999)) [only used in adamw optimizer]')
parser.add_argument('--optimizer_momentum', type=float, default=0, help='momentum factor (default: 0) '
                                                                        '[only used in sgd optimizer]')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight_decay for optimizer')

# ------------------------------------LR scheduler-------------------------------------------------------
parser.add_argument('--LR_scheduler', type=str, default='step', choices=['cosine', 'linear', 'step'],
                    help='LR scheduler used to adjust lr, they all used lr warmup')
parser.add_argument('--warmup_lr', type=float, default=5e-7,
                    help='lr warmup begin with this warmup_lr, until increase to learning_rate')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='num of epochs to increase lr from warmup_lr to learning_rate')
parser.add_argument('--decay_epochs', type=int, default=10,
                    help='decay lr per decay_epochs [only used in step LR_scheduler]')
parser.add_argument('--decay_rate', type=float, default=0.9,
                    help='decay lr by multiply decay_rate [only used in step LR_scheduler]')
parser.add_argument('--min_lr', type=float, default=5e-7, help='min lr [only used in cosine LR_scheduler]')

# ----------------------------------------------------------------------------------------------------
# Model settings
# ----------------------------------------------------------------------------------------------------
parser.add_argument('--model_name', type=str, default='DanQ', help='model name')
parser.add_argument('--activate_type', type=str, default='relu',
                    choices=['relu', 'tanh', 'sigmoid', 'elu', 'LeakyReLU', 'Prelu'],
                    help='activate layer type in all model')

parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--save_complete_model', type=ast.literal_eval, default=False, help='save_complete_model')
parser.add_argument('--only_save_best', type=ast.literal_eval, default=True)
args = parser.parse_args()

import argparse, glob, os, warnings, time
from tools.tools import *
from trainer import *
from dataLoader import *

parser = argparse.ArgumentParser(description = "Target Speaker VAD")

### Training setting
parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--warm_up_epoch',  type=int, default=10,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=10,      help='Batch size')
parser.add_argument('--rs_len',     type=float, default=16,      help='Input length (second) of reference speech')
parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,        help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.0001,    help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.90,     help='Learning rate decay every [test_step] epochs')
parser.add_argument('--max_speaker',type=int, help='Maximum number of speakers')
### Testing setting
parser.add_argument('--test_shift', type=float, default=16,      help='Input shift (second) for testing')
parser.add_argument('--min_silence', type=float, default=0.32,      help='Remove the speech with short slience during testing')
parser.add_argument('--min_speech', type=float, default=0.00,      help='Combine the short speech during testing')
parser.add_argument('--threshold', type=float, default=0.50,      help='The threshold during testing')
parser.add_argument('--init_model',  type=str,   default="",  help='Init TS-VAD model from pretrain')

### Data path
parser.add_argument('--train_list', type=str,   default="data/alimeeting/Train_Ali_far/ts_Train.json",     help='The path of the training list')
parser.add_argument('--train_path', type=str,   default="data/alimeeting/Train_Ali_far", help='The path of the training data')
parser.add_argument('--eval_list',  type=str,   default="data/alimeeting/Eval_Ali_far/ts_Eval.json",      help='The path of the evaluation list')
parser.add_argument('--eval_path',  type=str,   default="data/alimeeting/Eval_Ali_far", help='The path of the evaluation data')
parser.add_argument('--save_path',  type=str,    default="", help='Path to save the clean list')
parser.add_argument('--musan_path',  type=str,   default="/workspace/TSVAD_pytorch/ts-vad/data/musan", help='The path of the evaluation data')
parser.add_argument('--rir_path',  type=str,   default="/workspace/TSVAD_pytorch/ts-vad/data/RIRS_NOISES/simulated_rirs", help='The path of the evaluation data')
parser.add_argument('--simtrain', type=bool, default=False, help='For simulated data training pass train_list and train_path of simulated data. Eval list and path is not used in this mode. You can call s.eval_network(args) if you wish to run eval as well.')

### Others
parser.add_argument('--speech_encoder_pretrain',  type=str,   default="pretrained_models/WavLM-Base+.pt",  help='Path of the pretrained speech_encoder')
parser.add_argument('--train',   dest='train', action='store_true', help='Do training')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Do evaluation')

## Init folders, trainer and loader
args = init_system(parser.parse_args())
s = init_trainer(args)
args = init_loader(args)

## Evaluate only
if args.eval == True:
	s.eval_network(args)
	quit()

## Training
if args.train == True:
	while args.epoch < args.max_epoch:
		for param in s.ts_vad.module.speech_encoder.parameters():
			if args.epoch < args.warm_up_epoch:
				param.requires_grad = False
			else:
				param.requires_grad = True
		args = init_loader(args) # Random the training list for more samples
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			s.save_parameters(args.model_save_path + "/model_%04d.model"%args.epoch)
			if not args.simtrain:
				s.eval_network(args)
		args.epoch += 1
	quit()

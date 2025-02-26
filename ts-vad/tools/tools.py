import os, numpy, torch, warnings, glob

def init_system(args):
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.score_save_path      = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    args.rttm_save_path     = os.path.join(args.save_path, 'res_rttm')
    os.makedirs(args.model_save_path, exist_ok = True)

    args.modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
    args.modelfiles.sort()
    args.score_file = open(args.score_save_path, "a+")
    return args


# remove the short silence
def change_zeros_to_ones(inputs, min_silence, threshold):
	res = []
	num_0 = 0
	thr = int(min_silence // 0.04)
	for i in inputs:
		if i >= threshold:
			if num_0 != 0:
				if num_0 > thr:
					res.extend([0] * num_0)
				else:
					res.extend([1] * num_0)
				num_0 = 0		
			res.extend([1])
		else:
			num_0 += 1
	if num_0 > thr:
		res.extend([0] * num_0)
	else:
		res.extend([1] * num_0)
	return res

# Combine the short speech segments
def change_ones_to_zeros(inputs, min_speech, threshold):
	res = []
	num_1 = 0
	thr = int(min_speech // 0.04)
	for i in inputs:
		if i < threshold:
			if num_1 != 0:
				if num_1 > thr:
					res.extend([1] * num_1)
				else:
					res.extend([0] * num_1)
				num_1 = 0		
			res.extend([0])
		else:
			num_1 += 1
	if num_1 > thr:
		res.extend([1] * num_1)
	else:
		res.extend([0] * num_1)
	return res
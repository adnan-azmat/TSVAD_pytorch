import glob, numpy, os, random, soundfile, torch, json, wave
from collections import defaultdict
from scipy import signal

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	evalLoader = eval_loader(**vars(args))
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args

class train_loader(object):
	def __init__(self, train_list, train_path, rs_len, musan_path, rir_path, simtrain, max_speaker, **kwargs):
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

		self.train_path = train_path
		self.simtrain = simtrain
		self.max_speaker = max_speaker
		self.rs_len = int(rs_len * 25) # Number of frames for reference speech

		self.data_list = []
		self.label_dic = defaultdict(list)
		self.speaker_to_utt = defaultdict(list)
		self.room_to_speaker = defaultdict(list)

		lines = open(train_list).read().splitlines()
		filename_set = set()
		# Load the data and labels
		for line in lines:
			# if line length is 0, skip
			if len(line) == 0:
				continue
			
			dict = json.loads(line)
			length = len(dict['labels']) # Number of frames (1s = 25 frames)
			filename = dict['filename']
			speaker_name = dict['speaker_id']
			labels = dict['labels']
			speaker_id = str(dict['speaker_key'])
			full_id = filename + '_' + speaker_id
			self.label_dic[full_id] = labels
			self.speaker_to_utt[speaker_name].append(full_id)
			self.room_to_speaker[filename].append(speaker_name)
			if filename in filename_set:
				pass
			else:
				filename_set.add(filename)
				start_place = random.randint(0, rs_len) * 25 # Random start time, to make more training samples
				for start in range(start_place, length - self.rs_len, self.rs_len):
					folder = self.train_path + '/target_audio/' + filename + '/*.wav'
					audios = glob.glob(folder)
					num_speaker = len(audios) - 1 # The total number of speakers, 2 or 3 or 4
					data_intro = [filename, num_speaker, start, start + self.rs_len]
					self.data_list.append(data_intro)
         
	def __getitem__(self, index):
		# T: number of frames (1s contrains 25 frames)
		# ref_speech : 16000 * T
		# labels : max_speaker, T
		# target_speech: max_speaker, 192
		file, num_speaker, start, stop = self.data_list[index]
		speaker_ids = self.get_ids(file, num_speaker)
		ref_speech, labels = self.load_rs(file, speaker_ids, start, stop)
		target_speech = self.load_ts(file, speaker_ids)
		return ref_speech, target_speech, labels
	
	def get_ids(self, file, num_speaker):
		max_speaker = self.max_speaker
		path = self.train_path + "/target_audio/" + file
		# get all the wav files in the path
		folder = path + '/*.wav'
		audios = glob.glob(folder)
		audios.remove(path + "/all.wav")
		audios = [k.split('/')[-1].split('.')[0] for k in audios]
		audios = [int(k) for k in audios]
		speaker_ids = audios

		speaker_ids = speaker_ids[:max_speaker]
		while len(speaker_ids) < max_speaker:
			speaker_ids.append(0)
		
		random.shuffle(speaker_ids)
		return speaker_ids
	
	def load_rs(self, file, speaker_ids, start, stop):
		ref_speech, _ = soundfile.read(self.train_path + '/target_audio/' + file + '/all.wav', start = start * 640, stop = stop * 640 + 240) # Since 25 * 640 = 16000
		if len(ref_speech.shape) > 1:
			ref_speech = ref_speech[:, 0] 
		
		frame_len = int(self.rs_len / 25 * 16000) + 240
		ref_speech = numpy.expand_dims(numpy.array(ref_speech), axis = 0)
		augtype = random.randint(0,2)
		if augtype == 0:
			ref_speech = ref_speech
		elif augtype == 1:
			ref_speech = self.add_rev(ref_speech, length = frame_len)
		elif augtype == 2:
			ref_speech = self.add_noise(ref_speech, 'noise', length = frame_len)
		ref_speech = ref_speech[0]
		
		ref_speech = torch.FloatTensor(numpy.array(ref_speech))
		labels = []
		for speaker_id in speaker_ids:
			if speaker_id != 0:
				full_label_id = file + '_' + str(speaker_id)
				label = self.label_dic[full_label_id]
				labels.append(label[start:stop]) # Obatin the labels for the reference speech
			else:
				label = [0] * (stop - start)
				labels.append(label)
		labels = numpy.array(labels) # max_speaker, T
		return ref_speech, labels
	
	def load_ts(self, file, speaker_ids):
		target_speeches = []
		for speaker_id in speaker_ids:
			if speaker_id != 0:
				path = self.train_path + '/target_embedding/' + file + '/' + str(speaker_id) + '.pt'
			else:
				speakers_in_this_videos = self.room_to_speaker[file]
				candidate_speakers = [k for k in self.speaker_to_utt.keys() if k not in speakers_in_this_videos]
				random_speaker = random.choice(candidate_speakers)
				random_file = random.choice(self.speaker_to_utt[random_speaker])
				prefix = random_file[:random_file.rfind('_')]
				path = self.train_path + '/target_embedding/' + prefix + '/' + str(random_file.split('_')[-1]) + '.pt'
			
			feature = torch.load(path, map_location=torch.device('cpu'))
			feature = feature[random.randint(0,feature.shape[0]-1),:]
			target_speeches.append(feature)
		target_speeches = torch.stack(target_speeches) # max_speaker, 192
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, _     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
	
class eval_loader(object):
	def __init__(self, eval_list, eval_path, rs_len, test_shift, max_speaker, **kwargs):
		self.eval_path = eval_path
		self.rs_len = int(rs_len * 25)

		self.data_list = []
		self.label_dic = defaultdict(list)
		self.speaker_to_utt = defaultdict(list)
		self.room_to_speaker = defaultdict(list)
		self.max_speaker = max_speaker

		lines = open(eval_list).read().splitlines()
		filename_set = set()
		for line in lines:
			dict = json.loads(line)
			length = len(dict['labels']) # Number of frames (1s = 25 frames)
			filename = dict['filename']
			speaker_name = dict['speaker_id']
			labels = dict['labels']
			speaker_id = str(dict['speaker_key'])
			full_id = filename + '_' + speaker_id
			self.label_dic[full_id] = labels
			self.speaker_to_utt[speaker_name].append(full_id)
			self.room_to_speaker[filename].append(speaker_name)
			if filename in filename_set:
				pass
			else:
				filename_set.add(filename)
				dis = int(test_shift * 25)
				for start in range(0, length - self.rs_len, dis):
					folder = self.eval_path + '/target_audio/' + filename + '/*.wav'
					audios = glob.glob(folder)
					num_speaker = len(audios) - 1
					data_intro = [filename, num_speaker, start, start + self.rs_len]
					self.data_list.append(data_intro)

	def __getitem__(self, index):
		file, num_speaker, start, stop = self.data_list[index]
		speaker_ids = self.get_ids(file, num_speaker)
		ref_speech, labels = self.load_rs(file, speaker_ids, start, stop)
		target_speech = self.load_ts(file, speaker_ids)
		return ref_speech, target_speech, labels, file, numpy.array(speaker_ids), numpy.array(start)
	
	def get_ids(self, file, num_speaker):
		path = self.eval_path + "/target_audio/" + file
		max_speaker = self.max_speaker
		# get all the wav files in the path
		folder = path + '/*.wav'
		audios = glob.glob(folder)
		audios.remove(path + "/all.wav")
		audios = [k.split('/')[-1].split('.')[0] for k in audios]
		audios = [int(k) for k in audios]
		speaker_ids = audios

		speaker_ids = speaker_ids[:max_speaker]
		while len(speaker_ids) < max_speaker:
			speaker_ids.append(0)
		
		return speaker_ids
	
	def load_rs(self, file, speaker_ids, start, stop):
		ref_speech, _ = soundfile.read(self.eval_path + '/target_audio/' + file + '/all.wav', start = start * 640, stop = stop * 640 + 240)		
		if len(ref_speech.shape) > 1:
			ref_speech = ref_speech[:, 0]
		ref_speech = torch.FloatTensor(numpy.array(ref_speech))
		labels = []
		for speaker_id in speaker_ids:
			if speaker_id != 0:
				full_label_id = file + '_' + str(speaker_id)
				label = self.label_dic[full_label_id]
				label_here = label[start:stop]
				if len(label_here) < stop - start:
					label_here = [0] * (stop - start)
				labels.append(label_here)
			else:
				label = [0] * (stop - start)
				labels.append(label)
			
		labels = numpy.array(labels)
		return ref_speech, labels
	
	def load_ts(self, file, speaker_ids):
		target_speeches = []
		for speaker_id in speaker_ids:
			if speaker_id != 0:
				path = self.eval_path + '/target_embedding/' + file + '/' + str(speaker_id) + '.pt'
			else:
				speakers_in_this_videos = self.room_to_speaker[file]
				candidate_speakers = [k for k in self.speaker_to_utt.keys() if k not in speakers_in_this_videos]
				random_speaker = ""
				
				if len(candidate_speakers) == 0:
					random_speaker = random.choice(self.room_to_speaker[file])
				else:
					random_speaker = random.choice(candidate_speakers)
				
				random_file = random.choice(self.speaker_to_utt[random_speaker])

				# create variable prefix, which is the substring of random_file from 0 to last occurence of _
				prefix = random_file[:random_file.rfind('_')]

				path = self.eval_path + '/target_embedding/' + prefix + '/' + str(random_file.split('_')[-1]) + '.pt'
			feature = torch.load(path, map_location=torch.device('cpu'))
			feature = torch.mean(feature, dim = 0)
			target_speeches.append(feature)
		target_speeches = torch.stack(target_speeches)
		return target_speeches
	
	def __len__(self):
		return len(self.data_list)


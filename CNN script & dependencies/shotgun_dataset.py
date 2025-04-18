import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

#makes spectrogram and label mask
class shotgun_training_dataset(Dataset):
	def __init__(self, audio_path, labels_path, clip_len=4, max_time=-1, sample_rate=24000, n_fft=1024, hop_length=512, n_mels=128):
		self.audio_path = audio_path
		
		#get labels
		labels = pd.read_csv(labels_path, sep='\t', header=None, names=['start_time', 'end_time', 'label'])

		#drop anything after a max time
		if max_time != -1:
			labels = labels.loc[labels['start_time'] < max_time]
			
		self.clips = {}

		#list of starting times
		if max_time == -1:
			audio_len = int(librosa.get_duration(path=audio_path))
		else:
			audio_len = max_time
		self.time_starts = [i*clip_len for i in range(audio_len//clip_len)]

		#add to dictionary based on time_start indexes
		for i in range(len(self.time_starts) - 1):
			clip_labels = labels.loc[(labels['start_time'] > self.time_starts[i]) & (labels['start_time'] <= self.time_starts[i+1])]
			if not clip_labels.empty:
				#append labels for clip
				self.clips[i] = clip_labels

		#info on how to read audio file
		self.sr = sample_rate
		self.clip_len = clip_len
		self.hop_length = hop_length
		self.n_fft = n_fft
		
	#get length of dataset
	def __len__(self):
		return len(self.time_starts)

	def __getitem__(self, idx):
		#get clip
		start_time = self.time_starts[idx]
		#get labels
		if idx in self.clips:#get datafram label from dictionary
			labels = self.clips[idx]
		else:
			labels = [] #empty label

		#make spctrogram
		data, sr = librosa.load(self.audio_path, sr=self.sr, offset=start_time, duration=self.clip_len)
		mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
		mel_spec = librosa.power_to_db(mel_spec)
		mel_spec /= 80 #normalize
		mel_spec = torch.tensor(mel_spec, dtype=torch.float32) #convert to tensor
		mel_spec = mel_spec.unsqueeze(0) #add single channel dimension
		
		#make label array
		label_mask = np.zeros(188)
		offsets = np.arange(0,188) * 4/188

		#check if empty list
		if len(labels) != 0:
			#fill if not empty
			for index, row in labels.iterrows():
				start_idx = np.absolute(offsets - (row['start_time'] % 4)).argmin()
				end_idx = np.absolute(offsets - (row['end_time'] % 4)).argmin()

				label_mask[start_idx:end_idx] = 1

		label_mask = torch.tensor(label_mask, dtype=torch.float32)#convert to tensor

		return mel_spec, label_mask

#makes only the spectrogram
class shotgun_inference_dataset(Dataset):
	def __init__(self, audio_path, clip_len=4, sample_rate=24000, n_fft=1024, hop_length=512, n_mels=128):
		self.audio_path = audio_path

		audio_len = int(librosa.get_duration(path=audio_path))
		self.num_clips = audio_len // clip_len
		
		self.clips = [i*clip_len for i in range(self.num_clips)]

		self.sr = sample_rate
		self.clip_len = clip_len
		self.hop_length = hop_length
		self.n_fft = n_fft

	def __len__(self):
		return self.num_clips

	def __getitem__(self, idx):
		start_time = self.clips[idx]

		#make spctrogram
		data, sr = librosa.load(self.audio_path, sr=self.sr, offset=start_time, duration=self.clip_len)
		mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
		mel_spec = librosa.power_to_db(mel_spec)
		mel_spec /= 80 #normalize
		mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
		mel_spec = mel_spec.unsqueeze(0)

		return mel_spec

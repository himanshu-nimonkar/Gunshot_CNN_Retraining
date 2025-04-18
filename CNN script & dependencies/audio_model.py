import torch
import torch.nn as nn
import torchaudio


class shotgun_detector(torch.nn.Module):

	def __init__(self, input_shape):
		super().__init__()
		#tensor input dimensions:
		#batch size, channels, frequency, time
		self.encoding_block = nn.Sequential(
			nn.Conv2d(1, 128, kernel_size=(3,3), padding=1),
			nn.MaxPool2d(kernel_size=(8,1)), #only pool the frequency dimension (8,1), not the time dimension
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
			nn.MaxPool2d(kernel_size=(2,1)), #only pool the frequency dimension (8,1), not the time dimension
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
			nn.MaxPool2d(kernel_size=(2,1)), #only pool the frequency dimension (8,1), not the time dimension
			nn.ReLU(),
		)

		#LSTM input dimensions:
		#batch, feature, time
		#output is a tensor where each timestep has a feature set made from future and past time steps
		final_shape = input_shape[0] // 32
		self.recurrent_layer = nn.LSTM(128 * final_shape, 32, num_layers=2, batch_first=True, bidirectional=True)

		#output layer
		self.classification_layer = nn.Linear(64, 1)

	def forward(self, x):
		#run through convolutional layers
		x = self.encoding_block(x)
		x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
		x = x.transpose(1, 2) #n, l, f
		
		#send extracted features to recurrent layer
		x, states= self.recurrent_layer(x)
		predictions=[]
		for i in range(x.shape[1]): #make a prediction for each timestep
			out = x[:, i, :]
			out = self.classification_layer(out)
			predictions.append(out)

		#convert to tensor
		x = torch.cat(predictions, dim=1)
		
		return x
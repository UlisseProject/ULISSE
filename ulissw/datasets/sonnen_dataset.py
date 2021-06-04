import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class CustomerDataset(Dataset):
	def __init__(self, folder, prediction_interval):
		if not isinstance(prediction_interval, tuple):
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		if len(prediction_interval) != 2:
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		self.db = torch.tensor(CustomerDataset.load_df(folder), dtype=torch.float32)
		self.input_size = prediction_interval[0]
		self.output_size = prediction_interval[1]
		self.n_seqs = self.db.shape[0]
		self.len = np.prod(self.db.shape) // self.input_size - self.output_size

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		user = self.db[index % self.n_seqs]
		seq_num = index // self.n_seqs
		start_seq = seq_num*self.input_size
		seq = user[start_seq : start_seq + self.input_size]
		label = user[start_seq + self.input_size : start_seq + self.input_size + self.output_size]

		return seq, label 

	@staticmethod
	def load_df(folder):
		db = []
		right_length = 24*365*4

		for user_file in glob.glob(folder+"/*/*"):
		    df = CustomerDataset.read_cust_df(user_file)
		    if df.shape[0] == right_length:
			    db.append(df.cons.to_numpy(dtype=np.float64))		
		return np.stack(db)

	@staticmethod
	def read_cust_df(file):
		df = pd.read_csv(file, header=None, usecols=[1,2,3])
		df.columns =["timestamp", "pv", "cons"]
		df.timestamp = pd.to_datetime(df.timestamp)
		df.sort_values(by='timestamp', ignore_index=True)

		return df
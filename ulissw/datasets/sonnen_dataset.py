import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, Subset


class CustomerDataset(Dataset):
	def __init__(self, folder, prediction_interval):
		if not isinstance(prediction_interval, tuple):
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		if len(prediction_interval) != 2:
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		self.dfs, self.db = CustomerDataset.load_df(folder)
		self.db = torch.tensor(self.db, dtype=torch.float32)
		self.input_size = prediction_interval[0]
		self.output_size = prediction_interval[1]
		self.n_seqs = self.db.shape[0]
		self.seq_len = self.db.shape[1]
		#self.len = self.n_seqs*( self.seq_len//self.input_size - (self.output_size // self.input_size + 1))
		self.sub_seq_per_seq = self.seq_len - (self.input_size + self.output_size)
		self.len = self.n_seqs*self.sub_seq_per_seq

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		user = self.db[index // self.sub_seq_per_seq]

		start_seq = index % self.sub_seq_per_seq
		seq = user[start_seq : start_seq + self.input_size]
		label = user[start_seq + self.input_size : start_seq + self.input_size + self.output_size]

		return seq, label 

	def train_test_split(self, after_date='2019-01-01'):
		# TODO: implement
		try:
			timestamp = pd.to_datetime(after_date)
		except Exception:
			raise ValueError("Pass a date in a suitable format to be parsed")

		return self, self

	@staticmethod
	def load_df(folder):
		db = []
		dfs = []
		right_length = 24*365*4

		for user_file in glob.glob(folder+"/*/*"):
		    df = CustomerDataset.read_cust_df(user_file)
		    if df.shape[0] == right_length:
			    dfs.append(df)
			    db.append(df.cons.to_numpy(dtype=np.float64))		
		return dfs, np.stack(db)

	@staticmethod
	def read_cust_df(file):
		df = pd.read_csv(file, header=None, usecols=[1,2,3])
		df.columns =["timestamp", "pv", "cons"]
		df.timestamp = pd.to_datetime(df.timestamp)
		df.sort_values(by='timestamp', ignore_index=True)

		return df
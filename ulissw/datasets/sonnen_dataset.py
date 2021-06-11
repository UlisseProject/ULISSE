import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, Subset


class CustomerDataset(Dataset):
	def __init__(self, folder, prediction_interval, strategy='all'):
		if not isinstance(prediction_interval, tuple):
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		if len(prediction_interval) != 2:
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		self.preprocess_info = {}
		self.sampling_strategy = strategy
		self.dfs, self.db = CustomerDataset.load_df(folder)
		self.db = torch.tensor(self.db, dtype=torch.float32)
		self.input_size = prediction_interval[0]
		self.output_size = prediction_interval[1]
		
		self.db = self.__format_db_for_strategy(self.db, self.sampling_strategy)
		self.__set_metadata()

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		user = self.db[index // self.sub_seq_per_seq]

		start_seq = index % self.sub_seq_per_seq
		seq = user[start_seq : start_seq + self.input_size]
		label = user[start_seq + self.input_size : start_seq + self.input_size + self.output_size]

		return seq, label 

	def __set_metadata(self):
		self.n_seqs = self.db.shape[0]
		self.seq_len = self.db.shape[1]
		self.sub_seq_per_seq = self.seq_len - (self.input_size + self.output_size)
		self.len = self.n_seqs*self.sub_seq_per_seq

	def train_test_split(self, after_date='2019-01-01'):
		# TODO: implement
		try:
			timestamp = pd.to_datetime(after_date)
		except Exception:
			raise ValueError("Pass a date in a suitable format to be parsed")

		return self, self

	def min_max(self):
		min_seq, max_seq = self.db.min(axis=1).values, self.db.max(axis=1).values
		self.preprocess_info['preprocessing'] = 'minmax'
		self.preprocess_info['min'] = min_seq
		self.preprocess_info['max'] = max_seq

		self.db = (self.db - min_seq) / (max_seq - min_seq)
	
	def revert_preprocessing(self, data):
		def revert_minmax(data):
			min_seq = self.preprocess_info['min']
			max_seq = self.preprocess_info['max'] 
			
			data = data*(max_seq - min_seq) + min_seq
			return data

		if 'preprocessing' in self.preprocess_info:
			reverser = locals()['revert_'+self.preprocess_info['preprocessing']]
			return reverser(data)
		return data
		
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

	@staticmethod
	def __format_db_for_strategy(db, strategy):
		def all(db):
			return db
		
		def sum(db):
			return db.sum(axis=0).unsqueeze(0)

		def mean(db):
			return db.mean(axis=0).unsqueeze(0)

		formatter = locals()[strategy]
		return formatter(db)

	@property
	def sampling_strategy(self):
	    return self.__sampling_strategy

	@sampling_strategy.setter
	def sampling_strategy(self, strategy):
		avail = ['all', 'sum', 'mean']

		if strategy not in avail:
			raise ValueError("The strategy you set is not supported")
		self.__sampling_strategy = strategy
		if hasattr(self, 'db'):
			self.db = self.__format_db_for_strategy(self.db, self.sampling_strategy)
			self.__set_metadata()

import glob
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, Subset


class CustomerDataset(Dataset):
	def __init__(self, folder, prediction_interval, strategy='all', add_month_hour=False):
		if not isinstance(prediction_interval, tuple):
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		if len(prediction_interval) != 2:
			raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
		self.preprocess_info = {}
		self.dfs, self.db = CustomerDataset.load_df(folder)
		self.db = torch.tensor(self.db, dtype=torch.float32)
		self.input_size = prediction_interval[0]
		self.output_size = prediction_interval[1]
		self.add_month_hour = add_month_hour
		self.sampling_strategy = strategy
		
		#self.db = self.__format_db_for_strategy(self.db, self.sampling_strategy)
		#self.__set_metadata()

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		user = self.db[index // self.sub_seq_per_seq]

		start_seq = index % self.sub_seq_per_seq
		if not self.add_month_hour:
			seq = user[start_seq : start_seq + self.input_size]
			label = user[start_seq + self.input_size : start_seq + self.input_size + self.output_size]
		else:
			seq = user[:, start_seq : start_seq + self.input_size]
			label = user[0, start_seq + self.input_size : start_seq + self.input_size + self.output_size]

		return seq, label 

	def __set_metadata(self):
		self.n_seqs = self.db.shape[0]
		if not self.add_month_hour:
			self.seq_len = self.db.shape[1]
		else:
			self.seq_len = self.db.shape[2]

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
		if not self.add_month_hour:
			min_seq = self.db.min() 
			max_seq = self.db.max()
		else:
			min_seq = self.db[:,0,:].min() 
			max_seq = self.db[:,0,:].max()
		self.preprocess_info['preprocessing'] = 'minmax'
		self.preprocess_info['min'] = min_seq
		self.preprocess_info['max'] = max_seq

		if not self.add_month_hour:
			self.db = ((self.db - min_seq) / (max_seq - min_seq))
		else:
			self.db[:,0,:] = ((self.db[:,0,:] - min_seq) / (max_seq - min_seq)).data
	
	def revert_preprocessing(self, data):
		def revert_month_hour(data):
			data[:, 0, :] = (data[:, 0, :] *11) + 1
			data[:, 1, :] = (data[:, 1, :] *3)
			return data

		def revert_minmax(data):

			min_seq = self.preprocess_info['min']
			max_seq = self.preprocess_info['max'] 
			
			data = data*(max_seq - min_seq) + min_seq
			return data

		if 'preprocessing' in self.preprocess_info:
			had_hour_month = False
			if len(data.shape) > 2:
				had_hour_month = True
				hour_month_tensor = data[:, 1:, :]
				data = data[:, 0, :]

			reverser = locals()['revert_'+self.preprocess_info['preprocessing']]
			
			if had_hour_month:
				data = reverser(data)
				data = data.unsqueeze(1).expand(-1, 3, -1).clone()
				data[:, 1:, :] = revert_month_hour(hour_month_tensor)

				return data
			else:
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

	def sample_df_cons(self, group_n=20):
		df_idxs = np.random.choice(len(self.dfs), group_n, replace=False)
		tot_cons = pd.Series(np.zeros(self.dfs[0].shape[0]))
		for idx in df_idxs:
			tot_cons += self.dfs[idx].cons

		return tot_cons

	def get_timestamp_filter(self, year=None):
		if year is None:
			years = np.unique(np.array(list(map(lambda df: df.iloc[0].timestamp.timetuple().tm_year, self.dfs))))
			year = random.choice(years)

		year_dfs = list(filter(lambda df: year ==  df.iloc[0].timestamp.timetuple().tm_year, self.dfs))
		return year_dfs[0]

	@staticmethod
	def __format_db_for_strategy(db, strategy, n, **kwargs):
		def all(db, **kwargs):
			return db
		
		def sum(db, **kwargs):
			return db.sum(axis=0).unsqueeze(0)

		def mean(db, **kwargs):
			return db.mean(axis=0).unsqueeze(0)

		def group_sum(db, n):
			return db[:-(db.shape[0] % n)].view(-1, n, db.shape[-1]).sum(axis=1)

		def group_avg(db, n):
			return db[:-(d.db.shape[0] % n)].view(-1, n, db.shape[-1]).mean(axis=1)

		formatter = locals()[strategy]
		res = formatter(db, n=n)
		if (kwargs.get('add_month_hour', False)):
			res = CustomerDataset.__append_month_hour(res, kwargs['timestamps'])
		return res

	@staticmethod
	def __append_month_hour(db, df_timestamps):
		df_timestamps['month'], df_timestamps['hour'] = zip(*df_timestamps.apply(
						lambda x: (x.timestamp.month, x.timestamp.hour), axis=1))
		df_timestamps['hour'] = df_timestamps['hour'].map(lambda x: x // 6).map(lambda x : x /3)
		df_timestamps['month'] = df_timestamps['month'].map(lambda x : (x-1) /11)
		
		hour_month_tensor = torch.tensor(df_timestamps[['month','hour']].to_numpy(dtype=np.float32)).transpose(0, 1)
		db = db.unsqueeze(1).expand(-1, 3, -1).clone()
		db[:, 1:, :] = hour_month_tensor

		return db.contiguous()

	@property
	def sampling_strategy(self):
	    return self.__sampling_strategy

	@sampling_strategy.setter
	def sampling_strategy(self, strategy):
		avail = ['all', 'sum', 'mean', 'group_mean', 'group_sum']
		n = -1

		if strategy.startswith('group'):
			try:
				n = int(strategy[len('group'):strategy.find('_')])
			except ValueError:
				raise ValueError('wrong strategy format')

			if strategy.endswith('sum'):
				strategy = strategy[:len('group')] + \
						   strategy[strategy.find('sum')-1:]
			elif strategy.endswith('mean'):
				strategy = strategy[:len('group')] + \
						   strategy[strategy.find('mean')-1:]
			
		if strategy not in avail:
			raise ValueError('The strategy you set is not supported')
		self.__sampling_strategy = strategy
		if hasattr(self, 'db'):
			self.db = self.__format_db_for_strategy(self.db, self.sampling_strategy, n, 
													add_month_hour=self.add_month_hour, timestamps=self.dfs[0])
			self.__set_metadata()

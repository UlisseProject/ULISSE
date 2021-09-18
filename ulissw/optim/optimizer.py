import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from ..models import TCN
from ..datasets import CustomerDataset
from ..utils import RangeDict, get_band_price, predict_sequences


class PVBoptimizer:
	bands = {}
	daily_slots = {
		0: np.arange(5*4, 13*4),
		1: np.arange(13*4, 21*4),
		2: np.arange(21*4, 24*4 + 5*4)
	}
	def __init__(self, pv, battery, prices_path, n_panels, 
				model_path, dataset_path, price_delta_inflate=0):
		self.ulisse_handler = {
			0 : self.handle_night_slot,
			1 : self.handle_day_slot,
			2 : self.handle_day_slot,
		}
		self.standard_handler = {
			0 : PVBoptimizer.standard_battery_behavior,
			1 : PVBoptimizer.standard_battery_behavior,
			2 : PVBoptimizer.standard_battery_behavior,
		}
		self.__setup_bands()
		self.__setup_dataset(dataset_path)
		self.__setup_model(model_path)

		self.prices_df = prices_path
		sampled_houses =  self.ds.sample_df_cons(group_n=20)
		production_data = pv.data
		self.in_data = self.__join_cons_prices(production_data, sampled_houses, n_panels)
		self.day_len = int(self.in_data.shape[0] / 365)
		self.battery = battery
		self.price_delta_inflate = price_delta_inflate
		
		self.tcn = None
		self.cons = None
		self.ds = None
		self.log_results = {}

	@property
	def prices_df(self):
		return self.__prices_df

	@prices_df.setter
	def prices_df(self, path):
		# data/bands_prices.csv
		if not os.path.isfile(path):
			raise Exception(f"The path {path} does not exist")
		
		df = pd.read_csv(path, delimiter=",", header=None, index_col=0)
		df = df.transpose()
		columns = ['F1', 'F2', 'F3'] 
		if not set(columns).issubset(set(df.columns)):
			raise ValueError(f"The dataframe should contain : {columns}")
		if len(df) != 12:
			raise ValueError("The prices dataframe should have 1 value per month of each band")
		
		self.__prices_df = df
	
	def set_predictor_model(self, model_path):
		tcn = TCN(input_size=self.window[0], num_inputs=1, output_size=self.window[1],
				  num_blocks=2, n_channels=128)
		tcn.load_state_dict(torch.load(model_path))

		n = self.ds.db.shape[-1] // self.window[1] 
		offset = 0
		pred_seq, _ = predict_sequences(tcn, self.ds.db, self.window[0], self.window[1], n, offset)
		
		return tcn, pred_seq

	def __join_cons_prices(self, df, cons_data, n_panels):
		if ((not 'timestamp' in df.columns) and (not 'date_time' in df.columns)):
			raise ValueError(f"The dataframe should contain : 'timestamp' or 'date_time'")
		columns = ['p_prod', 'hour', 'month'] # 'kwh', 
		if 'timestamp' in df.columns:
			columns += ['timestamp']
		else:
			columns += ['date_time']
		if not set(columns).issubset(set(df.columns)):
			raise ValueError(f"The dataframe should contain : {columns}")
		if ((not 'timestamp' in df.columns) and (not 'date_time' in df.columns)):
			raise ValueError(f"The dataframe should contain : 'timestamp' or 'date_time'")


		df = df.loc[:, columns]
		df['price'] = df.apply(get_band_price, args=(self.bands, self.prices_df), axis=1)
		df['price'] /= 1e3 # convert to €/kWh
		df.drop(columns=['month', 'hour'], inplace=True) 
		
		if not 'timestamp' in df.columns:
			df['timestamp'] = pd.to_datetime(df['date_time'], dayfirst=True)
		df = df.sort_values(by=['timestamp'])
		
		diffs = (df.iloc[1:]['timestamp'].reset_index(drop=True) - df.iloc[:-1]['timestamp'].reset_index(drop=True)).reset_index(drop=True)		
		diffs.index = range(1, len(diffs)+1)
		diffs = diffs.apply(lambda x: x.total_seconds()/60/60)
		diffs[0] = 0.5

		df['e_prod'] = (df['p_prod']*diffs*n_panels)/1e3
		df['cons'] = cons_data
		return df

	def get_cons_prod_times_prices(self):
		all_cons = np.array(self.in_data.cons, dtype=np.float32)
		all_prod = np.array(self.in_data.e_prod, dtype=np.float32)
		all_prices = np.array(self.in_data.price, dtype=np.float32)
		all_times = self.in_data.timestamp

		return all_cons, all_prod, all_times, all_prices	

	def run_standard_behavior(self, days):
		if not hasattr(days, '__iter__'):
			days = [days]
		spent, charged, soc, all_cons, all_prod, all_times, all_prices = self.optimize(days, self.standard_handler)
		self.log_results['standard'] = {
			'input': (all_cons, all_prod, all_times, all_prices),
			'output': (spent, charged, soc)
		}
		return spent, charged, soc, all_cons, all_prod

	def run_ulisse(self, days):
		if not hasattr(days, '__iter__'):
			days = [days]
		print("[@optimizer] Optimizing strategy....")
		spent, charged, soc, all_cons, all_prod, all_times, all_prices = self.optimize(days, self.ulisse_handler)
		self.log_results['ulisse'] = {
			'input': (all_cons, all_prod, all_times, all_prices),
			'output': (spent, charged, soc)
		}
		return spent, charged, soc, all_cons, all_prod

	def optimize(self, days, handler):
		self.battery.reset()
		initial_charge = 0.1*self.battery.max_kwh
		self.battery.charge(self.battery.p_n, initial_charge*3.6e3/self.battery.p_n)

		all_cons, all_prod, all_times, all_prices = self.get_cons_prod_times_prices()
		charged = np.zeros(len(all_cons))
		soc = np.zeros(len(all_cons))
		average_monthly_deltas = self.get_average_monthly_deltas()
		spent = 0

		for day in days:
			for slot in self.daily_slots:
				slot_idxs = self.daily_slots[slot] + day*self.day_len
				
				cons = all_cons[slot_idxs]
				prod= all_prod[slot_idxs]
				prices = np.copy(all_prices[slot_idxs])
				prices += (prices - prices.min())*self.price_delta_inflate
				mean_gap = average_monthly_deltas[day]

				spent_step, charged[slot_idxs], soc[slot_idxs] = handler[slot](cons=cons, prod=prod, prices=prices, 
																			   mean_gap=mean_gap, battery=self.battery)
				spent += spent_step

		return spent, charged, soc, all_cons, all_prod, all_times, all_prices

	def handle_self_consume_slot(self, to_buy, prices):
		charged = np.zeros(len(to_buy))
		soc = np.zeros(len(to_buy))
		spent = 0
		timestep_len = 60*15
		
		for time_step in range(len(to_buy)):
			
			if to_buy[time_step] > 0:
				e_left = self.battery.discharge(to_buy[time_step]*3.6e3/timestep_len, timestep_len)
				spent += prices[time_step]*e_left
				charged[time_step] -= np.abs((to_buy[time_step] - e_left))
				
			elif to_buy[time_step] < 0:
				self.battery.charge(np.abs(to_buy[time_step])*3.6e3/timestep_len, timestep_len)
				charged[time_step] += np.abs(to_buy[time_step])
			
			soc[time_step] = self.battery.soc
		to_buy -= to_buy
		
		return charged, to_buy, spent, soc    
                    
	def handle_night_slot(self, cons, prod, prices, mean_gap, **kwargs):
		spent = 0
		charged = np.zeros(len(cons))
		soc = np.zeros(len(cons))
		timestep_len = 60*15
		to_buy = cons - prod
		
		if to_buy.sum() < 0 or to_buy.sum() < self.battery.max_load():
			self.buy_estimated_gap(mean_gap + + 0.1*self.battery.max_kwh, 0, charged, soc)
			step_charge, to_buy, step_spent, soc = self.handle_self_consume_slot(to_buy, prices)
			
			spent += step_spent
			charged += step_charge
			
			return spent, charged, soc
		
		gap_to_buy = to_buy.sum() - self.battery.max_load() + 0.1*self.battery.max_kwh
		if self.battery.free_storage() < gap_to_buy:
			buy = self.battery.free_storage() 
		else:
			buy = gap_to_buy
		
		spent += prices.min()*(buy+ mean_gap)
		idxs_left = self.charge_battery_over_steps(buy+mean_gap, timestep_len, 0, charged, soc)
		
		#idxs_left = timeslot_charging+1
		step_charge, to_buy_step, step_spent, soc_step = self.handle_self_consume_slot(to_buy[idxs_left:], prices[idxs_left:])
		spent += step_spent
		charged[idxs_left:] += step_charge
		to_buy[idxs_left:] = to_buy_step
		soc[idxs_left:] = soc_step

		return spent, charged, soc

	def handle_day_slot(self, cons, prod, prices, **kwargs):
		spent = 0
		charged = np.zeros(len(cons))
		soc = np.zeros(len(cons))
		to_buy = cons - prod
		
		step_charge, to_buy_step, step_spent, soc_step = self.handle_self_consume_slot(to_buy, prices)
		spent += step_spent
		charged += step_charge
		to_buy = to_buy_step
		soc = soc_step
	
		return spent, charged, soc

	def charge_battery_over_steps(self, buy, timestep_len, start_time_step, charged, soc):
		# how many slots : enery that you need divided by energy that you can store in a slot
		timeslot_charging = int(np.floor(buy*3.6e3 / (self.battery.p_n * timestep_len)))
		if timeslot_charging > 0:
			last_slot_amount = buy*3.6e3 % (self.battery.p_n * timestep_len)
			slot_amount = (buy*3.6e3-last_slot_amount)/timeslot_charging
			
			for t in range(start_time_step, start_time_step + timeslot_charging):
				charged[t] += slot_amount
				self.battery.charge(self.battery.p_n, slot_amount/self.battery.p_n)
				soc[t] = self.battery.soc 
			charged[t+1] += last_slot_amount
			self.battery.charge(self.battery.p_n, last_slot_amount/self.battery.p_n)
			soc[t+1] = self.battery.soc
			
			return t+2
		else:
			slot_amount = buy*3.6e3
			t = start_time_step
			charged[t] += slot_amount
			self.battery.charge(self.battery.p_n, slot_amount/self.battery.p_n)
			soc[t] = self.battery.soc
		
			return t+1

	def buy_estimated_gap(self, mean_gap, start_time_step, charged, soc):
		if self.battery.free_storage() < mean_gap:
			buy = self.battery.free_storage()
		else:
			buy = mean_gap
		_ = self.charge_battery_over_steps(buy, 60*15, start_time_step, charged, soc)

	def plot_outcomes(self, behavior='standard', days=0, save=''):
		if not hasattr(days, '__iter__'):
			days = [days]
		self.plot_cons_prod_soc(self.log_results[behavior]['input'],
								self.log_results[behavior]['output'],
								days,
								self.daily_slots,
								save)

	def get_average_monthly_deltas(self):
		all_cons, all_prod, _, _ = self.get_cons_prod_times_prices()

		pricey_slots = np.concatenate((self.daily_slots[1], self.daily_slots[2]))
		pricey_band_idxs = np.array([pricey_slots + day*self.day_len for day in range(365)])
		pricey_band_idxs[-1][-20:] %= 35040 
		pricey_band_cons = all_cons[pricey_band_idxs]
		pricey_band_prod = all_prod[pricey_band_idxs]
		pricey_band_gap = pricey_band_cons - pricey_band_prod
		max_e_deltas = pricey_band_gap.sum(axis=1)

		running_mean = np.zeros(365) 
		for i in range(1, 29):
			running_mean[i] = max_e_deltas[:i].sum()/i
		running_mean[29:] = np.convolve(max_e_deltas, np.ones(30)/30, mode='valid')

		return running_mean

	@staticmethod
	def standard_battery_behavior(cons, prod, prices, battery, **kwargs):
		spent = 0
		to_buy = cons - prod
		soc = np.zeros(len(cons))
		charged = np.zeros(len(cons))
		time_slot = 60*15
		
		for i, buy in enumerate(to_buy):
			if buy < 0:
				p_charge = np.abs(buy*3.6e3)/time_slot
				battery.charge(p_charge, time_slot)
				charged[i] += np.abs(buy)
				
			elif buy > 0:
				e_left = battery.discharge(buy*3.6e3/time_slot, time_slot)
				spent += prices[i]*e_left
				charged[i] -= np.abs((buy - e_left))
			soc[i] = battery.soc
				
		return spent, charged, soc

	@staticmethod
	def plot_cons_prod_soc(inputs, output, days, daily_slots, save=''):
		cons, prod, timestamp, _ = inputs
		_, _, soc = output
		day_idxs = np.concatenate([
				np.concatenate([daily_slots[i] for i in daily_slots]) + day*24*4 for day in days 
			])
		
		timestamp = timestamp[day_idxs]
		fig, ax = plt.subplots(2, 1, figsize=(14,12))
		if len(days) == 1:
			x=timestamp.map(lambda x:str(x.timetuple().tm_hour) + ':' + str(x.timetuple().tm_min))
		else:
			day_idxs = day_idxs[::4]
			x=timestamp.map(lambda x:str(x.timetuple().tm_mday)+ '-' + str(x.month_name() + '\nh'+str(x.timetuple().tm_hour) ))[::4]

		cons = cons[day_idxs]
		prod = prod[day_idxs]
		soc = soc[day_idxs]

		ax[0].plot(x, cons, label="Consumed")
		ax[0].plot(x, prod, label="Produced")
		ax[0].grid()
		ax[0].legend()		
		l = len(cons)
		
		ax[1].plot(x, soc, label='SoC')
		if len(days) == 1:
			ax[1].set_xticks(np.arange(0, l, 4)) # ticks every hour
			ax[0].set_xticks(np.arange(0, l, 4))

			ax[1].set_xlabel(f'Hours of  the day;  {str(timestamp.iloc[0].timetuple().tm_mday)} of  {str(timestamp.iloc[0].month_name())}')
			ax[0].set_xlabel(f'Hours of  the day;  {str(timestamp.iloc[0].timetuple().tm_mday)} of  {str(timestamp.iloc[0].month_name())}')

		else:
			ax[1].set_xticks(np.arange(0, l, 6)) # ticks every 4 hours
			ax[0].set_xticks(np.arange(0, l, 6))

			ax[1].set_xlabel(f'Days from {str(timestamp.iloc[0].timetuple().tm_mday)} to {str(timestamp.iloc[-1].timetuple().tm_mday)} of {str(timestamp.iloc[0].month_name())}')
			ax[0].set_xlabel(f'Days from {str(timestamp.iloc[0].timetuple().tm_mday)} to {str(timestamp.iloc[-1].timetuple().tm_mday)} of {str(timestamp.iloc[0].month_name())}')

		ax[1].set_ylabel('Battery State Of Charge')
		ax[1].grid()
		ax[1].legend()
			
		ax[0].set_ylabel('[kWh]')
		if save != '':
			plt.savefig(save)
		plt.show()
	
	def __setup_dataset(self, dataset_path):
		if not os.path.isdir(dataset_path):
			raise ValueError('Dataset path does not exist')
		print("[@optimizer] Loading dataset....")
		self.window = (192, 16)
		self.ds = CustomerDataset(dataset_path, prediction_interval=(self.window[0], self.window[1]), strategy='group20_sum')
		self.ds.min_max()

	def __setup_model(self, model_path):
		print("[@optimizer] Predicting consumption....")
		if not os.path.isfile(model_path):
			raise ValueError('Model path does not exist')

		self.tcn, self.cons = self.set_predictor_model(model_path)

	@staticmethod
	def __setup_bands():
		if not isinstance(PVBoptimizer.bands, RangeDict):
			PVBoptimizer.bands = RangeDict()
			# workday
			PVBoptimizer.bands[range(0, 5)] = RangeDict()
			PVBoptimizer.bands[0][range(0, 7)] = 'F3'
			PVBoptimizer.bands[0][range(7, 8)] = 'F2'
			PVBoptimizer.bands[0][range(8, 19)] = 'F1'
			PVBoptimizer.bands[0][range(19, 23)] = 'F2'
			PVBoptimizer.bands[0][range(23, 24)] = 'F3'
			# saturday
			PVBoptimizer.bands[range(5, 6)] = RangeDict()
			PVBoptimizer.bands[5][range(0, 7)] = 'F3'
			PVBoptimizer.bands[5][range(7, 23)] = 'F2'
			PVBoptimizer.bands[5][range(23, 24)] = 'F3'
			# sunday/festive
			PVBoptimizer.bands[range(6, 7)] = RangeDict()
			PVBoptimizer.bands[6][range(0, 24)] = 'F3'
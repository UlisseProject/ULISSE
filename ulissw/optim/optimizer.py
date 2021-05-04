import pandas as pd
import os

from ..utils import RangeDict, get_band_price


class PVBoptimizer:
	bands = {}
	def __init__(self, cons_prod_df, prices_path, n_panels):
		self.__setup_bands()

		self.prices_df = prices_path
		self.in_data = self.__join_cons_prices(cons_prod_df, n_panels)

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

	def __join_cons_prices(self, df, n_panels):
		columns = ['date_time', 'kwh', 'p_prod', 'hour', 'month'] 
		if not set(columns).issubset(set(df.columns)):
			raise ValueError(f"The dataframe should contain : {columns}")
		
		df = df.loc[:, columns]
		df['price'] = df.apply(get_band_price, args=(self.bands, self.prices_df), axis=1)
		df.drop(columns=['month', 'hour'], inplace=True) 
		
		df['timestamp'] = pd.to_datetime(df['date_time'], dayfirst=True)
		df = df.sort_values(by=['timestamp'])
		
		diffs = (df.iloc[1:]['timestamp'].reset_index(drop=True) - df.iloc[:-1]['timestamp'].reset_index(drop=True)).reset_index(drop=True)		
		diffs.index = range(1, len(diffs)+1)
		diffs = diffs.apply(lambda x: x.total_seconds()/60/60)
		diffs[0] = 0.5

		df['e_prod'] = (df['p_prod']*diffs*n_panels)/1e3
		return df

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
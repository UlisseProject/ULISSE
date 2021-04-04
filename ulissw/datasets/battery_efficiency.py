import numpy as np
import pandas as pd


class EtaParser:
	def __init__(self, path=None):
		self.file_path = path
		self.X = None
		self.y = None
		self.p_in_dim = None
		self.soc_dim = None

	def parse_data(self):
		df = pd.read_csv(self.file_path, delimiter=",", header=None) 

		p_in = pd.DataFrame(df.iloc[1].dropna()[1:])
		soc = pd.DataFrame(df.iloc[2].dropna()[1:])
		self.p_in_dim = p_in.shape[0]
		self.soc_dim = soc.shape[0]

		self.X = p_in.merge(soc, how='cross').to_numpy(dtype=np.float64)
		self.y = df.iloc[0][1:].to_numpy(dtype=np.float64)

		return self.X, self.y

	def get_surface_mesh(self):
		p_in = self.X[:,0][::self.soc_dim]
		soc = self.X[:,1][:self.soc_dim]

		x_surf, y_surf = np.meshgrid(p_in, soc, indexing='ij')
		z = y.reshape(self.p_in_dim, self.soc_dim)

		return x_surf, y_surf, z
	
	def format(self, p_in, soc):
		if np.isscalar(p_in):
			p_in = np.expand_dims(np.array(p_in), 0)
		if np.isscalar(soc):
			soc = np.expand_dims(np.array(soc), 0)
		if p_in.shape != soc.shape:
			raise ValueError("Power e Soc data should be of the same size")
			
		X = np.stack((p_in, soc), axis=1)

		return X
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class EtaRegressor:
	def __init__(self, X, y, poly_degree=2):
		self.X = X
		self.y = y
		self.degre = poly_degree
		self.model = make_pipeline(PolynomialFeatures(poly_degree, include_bias=False), 
								   LinearRegression())

	def fit_data(self):
		self.model.fit(X, y)
		
		return self.model.score(X, y)

	def predict(self, X):
		y_pred = self.model(X)

		return y_pred

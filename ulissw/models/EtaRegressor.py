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
		self.model.fit(self.X, self.y)
		
		return self.model.score(self.X, self.y)

	def predict(self, X):
		y_pred = self.model.predict(X)
		if y_pred.shape[0] == 1:
			return y_pred[0]
		return y_pred

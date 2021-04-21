import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_surface(x_mesh, y_mesh, z_mesh, label=None, ax_names=['P in', 'SoC', 'Eta']):
	fig, ax = plt.subplots(figsize=(11,7), subplot_kw={"projection": "3d"})
	if label is None:
		surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cm.coolwarm,
                      		   linewidth=2, antialiased=False)
	else:
		surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cm.coolwarm,
                      		   linewidth=2, antialiased=False, label=label)
		plt.legend()
	#ax.set_zlim(.4, 1.)
	ax.zaxis.set_major_locator(LinearLocator(10))
	# A StrMethodFormatter is used automatically
	ax.zaxis.set_major_formatter('{x:.02f}')
	ax.set_xlabel(ax_names[0])
	ax.set_ylabel(ax_names[1])
	ax.set_zlabel(ax_names[2])

	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()


def plot_lineset(x_set, y_set, from_line=0, to_line=None, labels=None, legend_title='legend'):
	if to_line is None:
		to_line = x_set.shape[1]
	
	fig, ax = plt.subplots(figsize=(10, 7))
	plt.plot(x_set[:, from_line:to_line], y_set[:, from_line:to_line])

	if labels is not None:
		try:
			plt.legend(labels[from_line:to_line].round(1), title=legend_title)
		except:
			plt.legend(labels[from_line:to_line], title=legend_title)			
	plt.grid()
	plt.show()
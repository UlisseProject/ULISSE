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
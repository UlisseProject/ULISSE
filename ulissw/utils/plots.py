import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from .utils import get_filepath, predict_sequences


def plot_n_sequences(sequences, model, n, offset=0, windows=(192, 16), transformer=None, 
                     title="Demand estimation", save=None):
    if not isinstance(windows, tuple):
        raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
    if len(windows) != 2:
        raise ValueError("prediction_interval should be a tuple (input sequence len, predicted sequence len)")
    in_len = windows[0]
    out_len = windows[1]
    if sequences.shape[-1] < n + in_len + out_len:
        raise ValueError(f"The sequences you provided are not enough to evaluate {n} predictions")
    
    pred_seq, real_seq = predict_sequences(model, sequences, in_len, out_len, n, offset)
    
    if transformer is not None:
        pred_seq = transformer(pred_seq)
        real_seq = transformer(real_seq)
    
    fig, ax = plt.subplots(figsize=(10,8))
    plt.xlabel('hours')
    plt.ylabel('[kW]')
    plt.title(title)
    plt.plot(np.arange(n*out_len)/4, pred_seq, label="Predicted")
    plt.plot(np.arange(n*out_len)/4, real_seq, label="Real")
    plt.grid()
    plt.legend()
    if save is not None:
        plt.savefig(save)
    plt.show()                     


def plot_many(x=None, y=[], labels=[], save=False, fname='plot_many', autorange=True):
    ############## IGNORE ########
    if (y is None):
        raise ValueError("You should at least specify y, and it has to be a list")
    if not isinstance(y, list):
        y = [y]
    if fname.find(".") > 0:
        raise ValueError("In fname you should specify only the filename without extension.\nOnly"+ 
                         "extension supported is .eps and is generated automatically")
    if (x is None) and (sum([len(y_i) for y_i in y]) != len(y[0])*len(y)) :
        raise ValueError("x can be not specified only if all Ys have same length")
    elif x is None:
        x_i = range(len(y[0]))
    if labels != False:
        if not isinstance(labels, list):
            labels = [labels]
        if len(labels) < len(y):
            labels[len(labels):len(y)] = [f"curve n.{i+1}" for i in range(len(labels), len(y))]
    ################################
    
    fig, ax = plt.subplots(figsize=(11, 7)) # (11, 7) is the graph dimension in inches
    plt.xlabel("X label") # change axis label
    plt.ylabel("Y label")
    plt.title("Title") # and title
    
    for i, y_i in enumerate(y):
        if isinstance(x, list):
            x_i = x[i]
        if labels != False:
            plt.plot(x_i, y_i, label=labels[i]) # the string after label is displayed in the legend for the curve
        else:
            plt.plot(x_i, y_i)
            
    if autorange and isinstance(autorange, tuple):
        x_low = autorange[0][0] # to change the axis range setting these values below
        x_up = autorange[0][1]
        y_low = autorange[1][0]
        y_up = autorange[1][1]

        plt.xlim(x_low, x_up) # comment these 2 lines to get automatic 
        plt.ylim(y_low, y_up) # axis range based on data range
    plt.rcParams['font.size'] = 18 # change font size (axis labels, title, legend, ticks..)
    if labels != False:    
        plt.legend() # comment to remove legend
    plt.grid() # comment to remove grid
    
    ############## IGNORE ########
    if save:
        file_path = get_filepath(fname)
        plt.savefig(file_path, format='eps', dpi=None) # dpi can set the resolution in dots per inch
    plt.show()
    ########################


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
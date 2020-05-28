'''
make fake data with x number of gaussians for symbolic regression later
'''
import os
import numpy as np
np.random.seed(19) #20 okay as well except last 2 curves
import matplotlib.pyplot as plt


def one_gauss(x, peak, std, intensity, ybump):
	'''
	expect x to be a numpy array of locations

	https://en.wikipedia.org/wiki/Normal_distribution
	'''
	factor = 1/(std*np.sqrt(2*np.pi))
	inside_exp = -0.5 * np.square((x-peak)/std)
	return intensity*factor*np.exp(inside_exp) + ybump


def sum_gauss(x, peaks, stds, intensities, ybumps, plot=False, save_fig_dir=None):
	'''
	expect x to be a numpy array of locations
	'''
	if plot:
		fig = plt.figure(figsize=(20,10))

	x.sort() #helpful for plotting later
	y = np.zeros(shape=x.shape)
	for i, args in enumerate(zip(peaks, stds, intensities, ybumps)):
		one_y = one_gauss(x, *args)
		y += one_y
		if plot:
			plt.plot(x, one_y, linestyle='--', label="%i$^{th}$ gaussian" % i)

	# add noise
	wNoise = y + np.random.normal(size=y.shape)*3

	# plot the sum and the noise
	if plot:
		plt.plot(x, y, color='b', alpha=0.5, label="summed gaussians")
		plt.plot(x, wNoise, color='k', alpha=1.0, label="with noise")
		plt.legend()
		if (save_fig_dir is not None) and (os.path.isdir(save_fig_dir)):
			plt.savefig(os.path.join(save_fig_dir, "mixedgaussian.jpg"))
			plt.close()
		else:
			plt.show()

	return x, y, wNoise


def main(plot=False, save_fig_dir=None):
	gauss_count = 10
	xmin = 40
	xmax = 70
	xbuffer = (xmax-xmin)*0.1 # 10% of the domain on either side
	xpeaks = np.random.uniform(xmin+xbuffer, xmax-xbuffer, size=gauss_count) #xmin+xbuffer + np.random.random(gauss_count)*(xmax-xmin-2*xbuffer)
	stds = np.random.uniform(.3, 1, gauss_count) #(0,1) floats
	intensities = np.random.randint(10, 100, gauss_count)
	ybumps = np.zeros(gauss_count) #np.random.uniform(3, 10, gauss_count)
	# ^didn't make much sense in the end to add the bump for each curve because it's just adding a constant
	#each time so they just add up to a single big constant

	x_locations = np.arange(xmin, xmax, 0.02)

	xs, ys, noisy = sum_gauss(x_locations, xpeaks, stds, intensities, ybumps, plot, save_fig_dir)

	goal_features = np.vstack([xpeaks, stds, intensities, ybumps]).T #shape should be (gauss_count, 4)

	return xs, ys, noisy, goal_features


if __name__ == '__main__':
	'''
	run through command line for testing
	OR
	import for grabbing the data:
		import gp_analysis.fake_mixturegauss as fmg_data
		xs, ys, noisy, features = fmg_data.main(plot=True, savefig="./")
	'''
	xs, ys, noisy, features = main(plot=True, save_fig_dir=None)
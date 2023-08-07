#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file		 :Muti-station_volcanic_signals_classificaiton.py
@note		 :
@time		 :2020/12/05 10:54:53
@author		   :xcui
@version		:1.0
'''

import os
import shutil
import fnmatch
import sys
import re
import json
import glob
import math
import pickle
import numpy as np
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from matplotlib.widgets import Button
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


######################################################################################################################################
######################   FUNCTION   ##################################################################################################
######################################################################################################################################

# Function to handle the stop button click event
def quit_button_clicked(event):
    plt.close('all')
    raise SystemExit("Script terminated by user")



def SNR(data, arr, rate, win, tol=0.5):
	narr = int(arr * rate)
	nwin = int(win * rate)
	ntol = int(tol * rate)
	noise_e = np.sum(data[narr-nwin:narr-ntol]**2)+1e-15
	signal_e = np.sum(data[narr+ntol:narr+nwin]**2)+1e-15
	return 10*math.log10(signal_e/noise_e)


def amp_nmlz(fz,x):
	fz_nmlz = fz / fz[-1]
	x_int = np.sum(x) * (fz[1]-fz[0])
	x_nmlz = x / x_int *fz[-1]
	return x_nmlz, fz_nmlz


#sometimes the rate may not 100 , so it best for us to have a resample firstly.
#fft
def fft_amp(Data, rate, arr, win_len):

	narr = int(arr * rate)
	nwin_len = int(win_len * rate)
	Data = Data[narr-nwin_len:narr+nwin_len]
	sp = np.fft.fft(Data)
	freq = np.fft.fftfreq(len(Data), d=1.0/rate)
	amp = np.abs(sp)
	A_upper = np.mean(amp[(freq>5.0) & (freq<15.0)])
	A_lower = np.mean(amp[(freq>1.0) & (freq<5.0)])
	# input("wait  us...")
	return amp[(freq>=0) & (freq<15.0)], freq[(freq>=0) & (freq<15.0)], math.log10(A_upper/A_lower)


def picker(data, rate, nsta=10, nlta=100, uncer=0.5):


	stalta = recursive_sta_lta(data, nsta, nlta)
	n_max = np.argmax(stalta)
	quality = stalta[n_max]
	n_onset = int(uncer * rate)
	diff_max = np.diff(stalta[n_max-n_onset:n_max+n_onset])
	n_diff_max = np.argmax(np.diff(stalta[n_max-n_onset:n_max+n_onset]))
	n_pick = n_max - n_onset + n_diff_max
	arr = n_pick / rate
	return	arr, quality

def picker_new(file):
	dirname1 = os.path.dirname(file)
	num_event = os.path.basename(dirname1)
	dirname2 = os.path.dirname(dirname1)
	cat_in = [file for file in os.listdir(dirname2) if fnmatch.fnmatch(file, "CAT_IN" + '*')]
	if len(cat_in) != 1:
		return False
	cat_in = "{}/{}".format(dirname2, cat_in[0])
	df = pd.read_csv(cat_in, sep='\s+')

	arr_P = df.loc[df['EventID'] == int(num_event), 'arr_P[s]']
	if arr_P.empty:
		return False

	return arr_P.values[0]

def plot_array(array):
    plt.plot(array)
    plt.show()


	# input("wait...")

################################################################################################
def remove_dict(dict_event, dict_sta, namp_len):
	""" For each event, loop through each station and check if the amp_nmlz length is = 39
		If this number is not ok for a certain amount of station, remove the event from the list""" 
	need_delete = []
	for key in dict_event.keys():
		num_sta = 0
		for i in dict_sta.values():
			if (dict_event[key][i] != []):
				# remove those which length is not 39
				if len(dict_event[key][i][0])== namp_len:
					num_sta += 1
		if num_sta < least_station :
			need_delete.append(key)
			   
	for rem in range(len(need_delete)):
		dict_event.pop(need_delete[rem])
	print("the counts of events that meet the requirements are:\n"+str(len(dict_event.keys())))
	return dict_event


def get_median_value(dict_event, namp_len, whether_plot):
	""" We loop through each events from dict events, and for each each events we loop through each station
	and do the following:
	If data ara available for the station, we check in length of "Amp_nmls" is well = "namp_lem (=39), 
		--> if yes we add "Amp_nmlz" in tempo_amp and "fi" in tempo_fi
	Then, before looping to the next events we add in median_FI and median_amp the value from median station as follow:
		--> we check the legth of tempo_FI to know how many station do have value for this event
		--> if number is pair, we pop the tempo_FI array at index 0
		--> we calculte the index of median value of tempo_FI
		--> we add in median_FI and median_amp the value from tempo_xxx at median index.
	"""
	median_FI = []
	median_amp = []
	events_key = []
	waveform = []
	for key, value1 in dict_event.items():
		tempo_FI = []
		tempo_amp = []
		tempo_waveform = []
		# events_key.append(key)

		for para in dict_sta.values():
			if (value1[para] != []):
				if (len(value1[para][0]) == namp_len):
					# tempo_FI.append(value1[para][1])
					# tempo_amp.append(value1[para][0])
					# tempo_waveform.append(value1[para][2])
					median_FI.append(value1[para][1])
					median_amp.append(value1[para][0])
					waveform.append(value1[para][2])
					events_key.append(key)

		##################################################################################################
		####### Comment stuff below because not needed anymore since we work wuth unique station #########
		##################################################################################################


		# if len(tempo_FI)%2 == 1:
		# 	median_as = tempo_FI.index(np.median(tempo_FI))

		# 	if isinstance(median_as, int):
		# 		median_FI.append(tempo_FI[median_as])
		# 		median_amp.append(tempo_amp[median_as])
		# 		median_amp.append(tempo_amp[median_as])
		# 	else:
		# 		median_as = median_as[0]
		# 		median_FI.append(tempo_FI[median_as])
		# 		median_amp.append(tempo_amp[median_as])
		# 		waveform.append(tempo_waveform[median_as])
		# 		waveform[key] = [median_as, [tempo_waveform[median_as]]]
		# 		input("??????")
		# 	# print("median_FI = ", median_FI)
		# else:
		# 	# print("tempo_FI = ", tempo_FI)
		# 	tempo_FI.pop(0)
		# 	# print("tempo_FI = ", tempo_FI)
		# 	median_as = tempo_FI.index(np.median(tempo_FI))
		# 	if isinstance(median_as, int):
		# 		median_FI.append(tempo_FI[median_as])
		# 		median_amp.append(tempo_amp[median_as])
		# 	else:
		# 		median_as = median_as[0]
		# 		median_FI.append(tempo_FI[median_as])
		# 		median_amp.append(tempo_amp[median_as])
		# 		input("??????")
			# print("median_FI = ", median_FI)

		##################################################################################################



	bins_ = np.linspace(-1.5, 1, 30)
	plt.hist(median_FI, bins=bins_, edgecolor='k')
	plt.savefig('out/png/median_FI.png', format = 'png')
	pickle.dump(np.asarray(events_key), open('out/text/events_key.pkl', 'wb'))
	pickle.dump(np.asarray(median_amp), open('out/text/median_amp.pkl', 'wb'))
	if whether_plot:
		plt.show()
	plt.close()
	return median_FI, median_amp, events_key, waveform


def calculate_EM(median_amp):
	sum_amp = np.sum(median_amp[0]) * np.sqrt(2)
	median_distance = np.zeros((len(median_amp), len(median_amp)))
	for i in range(len(median_amp)):
		for j in range(i+1,len(median_amp)):
			median_distance[i, j] = np.sqrt(np.sum((median_amp[i][:]-median_amp[j][:])**2))/sum_amp
			median_distance[j, i] = median_distance[i, j]
			
	pickle.dump(np.asarray(median_distance), open('out/text/median_distance.pkl', 'wb'))
	return median_distance

######################################################################
"""
This part is clustering result display
"""
def clust_stats(clust, whether_plot):
	labels = np.unique(clust.labels_)
	n_clusters = len(labels)
	counts = np.zeros(n_clusters, dtype='int')
	for i in range(n_clusters):
		idx = np.where(clust.labels_ == labels[i])[0]
		counts[i] = len(idx)

	plt.figure()
	plt.bar(np.arange(n_clusters), counts)
	for i in range(n_clusters):
		plt.text(i, counts[i], str(counts[i]),
					horizontalalignment='center',
					verticalalignment='bottom')
	plt.xticks(np.arange(n_clusters))
	title = 'out/png/hist.png'
	plt.savefig(title, format='png')
	if whether_plot:
		plt.show()
	plt.close()


def new_catalog(labels, event_info, events_key, median_FI, sort_idx):
	k = 0
	event_info_temp = event_info[event_info['StartTime'].isin(events_key)]
	event_info_temp = event_info_temp.reset_index()
	new_events_catalog = []

	# event_info_temp.values[0][0] = str(event_info_temp.values[0][0])  \
	# 						  + ' FI' + ' CLUSTER'
	# add 2 new column to event catalog
	event_info_temp['zFI'] = False
	event_info_temp['zCLUSTER'] = False
	new_events_catalog.append(str(event_info_temp.keys()))
	for i in range(len(events_key)):
		print("1 - ", i)
		for j in range(k, len(event_info_temp.values)):
			
			# sys.stdout.write('\r' + f"Progress: {j}/{len(event_info_temp.values)}")
			# sys.stdout.flush()
			if events_key[i] == event_info_temp['StartTime'][j]:

				event_info_temp['zFI'][j] = str(round(median_FI[i], 2))
				event_info_temp['zCLUSTER'][j] = str(np.where(sort_idx == labels[i])[0][0])

				new_events_catalog.append(str(event_info_temp.values[j]))
				k = j
				break
	for i in range(len(sort_idx)):
		print("2 - ", i)
		idx_cls = np.where(labels == sort_idx[i])[0]
		with open('./out/text/cluster'+str(i)+'.dat', 'w') as f:
			for j in range(len(idx_cls)):
				f.write(new_events_catalog[idx_cls[j]] + '\n')

	event_info_temp.to_csv('out/text/new_catalog', index=False, header=True)
				 

def plot_rep(clust, amp, sort_idx, waveform, keys, noise_length, rate):
	labels = np.unique(clust.labels_)
	print("----- plot_rep -----")
	print(labels)
	num_rep = 100
	mx, my = 10, 10
	amp = np.array(amp)
	for label in labels:
		idx = np.where(clust.labels_ == sort_idx[label])[0]
		if len(idx)>num_rep:
			sel_idx = idx[np.random.choice(len(idx), num_rep)]
			# the num of every category <= 100 
		else:
			sel_idx = idx
		fig = plt.figure(figsize=(10, 10))
		for i in range(len(sel_idx)):
			plt.subplot(mx, my, i+1)
			plt.plot(amp[idx[i], :]/np.max(amp[idx[i], :]), c='k', linewidth=2)
			plt.axis('off')
		title1 = 'Cluster #' + str(label)
		title2 = ' ('+str(len(idx))+' members)'
		fig.suptitle(title1+title2, fontsize=40)
		title = 'out/png/cls'+str(label)+'.pdf'
		plt.savefig(title, format='pdf')
		plt.close()
	# plot waveform
	num_rep = 20
	mx, my = 20, 1
	n_pick = noise_length * rate
	n_text = int((len(waveform[0])/8)*7)
	wf = np.array(waveform)
	for label in labels:
		idx = np.where(clust.labels_ == sort_idx[label])[0]
		if len(idx)>num_rep:
			sel_idx = idx[np.random.choice(len(idx), num_rep)]
			# the num of every category <= 100 
		else:
			sel_idx = idx
		fig = plt.figure(figsize=(10, 10))
		for i in range(len(sel_idx)):
			plt.subplot(mx, my, i+1)
			# plt.plot(wf[idx[i], :]/np.max(wf[idx[i], :]), c='k', linewidth=2)
			plt.plot(wf[idx[i], :], c='k', linewidth=2)
			plt.text(n_text, 0, str(keys[idx[i]]), fontsize = 6,  bbox=dict(facecolor='lightgrey', alpha=0.9))
			# plt.annotate(str(keys[idx[i]]), fontsize = 6,  bbox=dict(facecolor='yellow', alpha=0.5))
			plt.axvline(x=n_pick, color='red', linestyle='--', linewidth=1)
			plt.axis('off')
		title1 = 'Waveform #' + str(label)
		title2 = ' ('+str(len(idx))+' members)'
		fig.suptitle(title1+title2, fontsize=40)
		title = 'out/png/wf'+str(label)+'.pdf'
		plt.savefig(title, format='pdf')
		plt.close()



#turn it to two function. 1.get the sirt_idx; 2. plot the median spectra
def freq_sort(labels, amp, n_cls):
	""" We loop loop through the number of cluster, and for each cluster we extract an array of the corresponding index number from labels
	which represents the event index. So we do have an array of event index which are in the same cluster = idx_cls
	We create an empty array with length = "Amp_nmlz"
	--> Then, we loop through idx_cls (= loop through evnets that are in the same cluster)
		- we caculate the mean_amp by additioning all amp_nmlz from event that are in the same cluster and we divide by number of events
		- we append this value in mean_amp_container (= mean of amp_nmlz for all clusters)
	We record in "max_amp_idx" for each cluster the index of highest 'amp' value 
	we record in sort_idx the sorted index value of max_amp_idx 
	"""
	amp = np.array(amp)
	mean_amp_container = []
	for i in range(n_cls):
		idx_cls = np.where(labels == i)[0]
		mean_amp = np.zeros(amp.shape[1])
		for j in idx_cls:
			mean_amp += amp[j,:]

		mean_amp = mean_amp / len(idx_cls)
		mean_amp_container.append(mean_amp)
	
	max_amp_idx = np.argmax(mean_amp_container, axis=-1)
	sort_idx = np.argsort(max_amp_idx)
	return np.array(mean_amp_container), sort_idx




def plot_mean_spectra(labels, amp, median_distance, mean_amp_container, sort_idx):
	""" We loop through each cluster (=len(sort_idx))
		- idx_cls = array where value are index of events for a same cluster
		- len_idx = number of events in a cluster
		- X_cls = filtered median_distance array (2d mirror array) and keep only events for this cluster
		- median_dist = median dist array from X_cls
		- i_min = index of min value fron median_dist
		- Plot in black the amp_nmlz of event from the cluster which has the minimum median_distance with other events from the same cluster
		- Plot in red the mean_amp_container value of the current cluster
	"""
	len_idx = []
	amp = np.array(amp)
	print("amp =")
	print(amp)
	plt.figure(figsize=(16, 10))
	for i in range(len(sort_idx)):
		# retrieve the cluster members
		idx_cls = np.where(labels == sort_idx[i])[0]
		len_idx.append(len(idx_cls))
		X_cls = median_distance[idx_cls, :][:, idx_cls]
		# find the reference smediantf (closest to the cls center)
		# i.e., the one with min  distance from other group members
		median_dist = np.median(X_cls, axis=1)
		imin = np.argmin(median_dist) # locade of the min 
		# obtain and plot the stretched stf relative to the reference
		ax = plt.subplot(4, 5, i+1)
		plt.plot(amp[idx_cls][imin], linewidth=2, c='#333333')	# black
		plt.plot(mean_amp_container[sort_idx[i]], linewidth=2, c='#FF3333') # red
		plt.xticks([])
		plt.yticks([])
		plt.axis('off')
		plt.text(0.01, 0.99, 'Cluster #'+str(i),
				fontsize=15, fontweight='bold',
				horizontalalignment='left',
				verticalalignment='center',
				transform=ax.transAxes)
		plt.text(0.99, 0.99, '('+str(len(idx_cls))+')',
				horizontalalignment='right',
				verticalalignment='center',
				fontsize=15, transform=ax.transAxes)
	plt.savefig('out/png/mean_spectra.pdf', format='pdf')
	if whether_plot:
		plt.show()
	plt.close()
	return len_idx


def freq_energy_distribution(mean_amp_container, freq, sort_idx, len_idx):
 # calculate features and plot
	max_amp = np.max(mean_amp_container[sort_idx], axis=-1)
	max_amp_idx = np.argmax(mean_amp_container[sort_idx], axis=-1)
	peak_freq = freq[max_amp_idx]
	plt.scatter(peak_freq, 1./max_amp, alpha = 0.75)
	for i in range(mean_amp_container.shape[0]):
		plt.text(peak_freq[i], 1./max_amp[i], str(i), size =8)
		plt.ylabel('Mean amp/Peak amp')
		plt.xlabel('Peak frequency /HZ')
	plt.savefig('out/png/fre_energy.pdf', format='pdf')
	if whether_plot:
		plt.show()
	plt.close()	  

	np.savetxt("out/text/peak_amp_size", list(zip(peak_freq, 1./max_amp, 0.1*np.log10(np.array(len_idx)), np.arange(20))))



def plot_matrix(median_distance):
	plt.figure(figsize=(10, 10))
	#imshow don't support float16
	im = plt.imshow(median_distance, origin='lower', cmap='RdBu',
			   vmin=0, vmax=0.25)		 
	#imshow don't support float16
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('Events', fontsize=25)
	plt.ylabel('Events', fontsize=25) 
	cb = plt.colorbar(im, ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25])
	cb.set_label('Dissimilarity', fontsize=25) 
	plt.savefig('out/png/med_dis.pdf', format = 'pdf')
	if whether_plot:
		plt.show()
	plt.close()


def plot_dendrogram(median_distance, n_cls):

	X_vec = squareform(median_distance) # The square matrices of vectors convert to each other
	linkage_matrix = linkage(X_vec, "complete")	 # Hierarchical clustering
	plt.figure(figsize=(20, 10))
	dendrogram(linkage_matrix, p=n_cls, truncate_mode="lastp")
	ax = plt.gca()
	ax.tick_params(axis='x', which='major', labelsize=20)
	ax.tick_params(axis='y', which='major', labelsize=20)
	# plt.ylabel('amp distance', fontsize= 40)
	# plt.title("Dendrogram with "+str(n_cls)+" clusters", fontsize=40)
	plt.savefig('out/png/dendrogram.pdf', format='pdf')
	if whether_plot:
		plt.show()
	plt.close()

def print_c(*args, sep=' ', end=''):
    output = sep.join(str(arg) for arg in args)
    print(output, end=end, flush=True)


def run_f(variable):
	""" decorator function which execute the function only if "variable" is true"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if variable:
                result = func(*args, **kwargs)
                return result
            # else:
            #     print("Function not executed because the variable is False.")
        return wrapper
    return decorator


######################################################################################################################################
######################   MAIN PROGRAM   ##############################################################################################
######################################################################################################################################


if __name__ == '__main__':


	#get para from config_json
	parser = ap.ArgumentParser(
		prog='Muti-station_volcanic_signals_classification.py',
		description='classified volcano signals')
	parser.add_argument('config_json')
	parser.add_argument('-P',default=False,action='store_true',help='Plot output')
	parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
	parser.add_argument("--stream", action="store_true", help="Display all streams plot")
	args = parser.parse_args()
	whether_plot = args.P
	verbose = args.verbose
	test_pick = args.stream

	@run_f(verbose)
	def print_x(*args):
		print(*args)

	@run_f(verbose)
	def input_x(*args):
		input(*args)



	print("Start scripts:")
	if verbose:
		print("Verbose mode is activated")
	print_c("Open config file --> ")

	# Open config file and write parameters into variables
	with open(args.config_json, "r") as f:
		params = json.load(f)

	# make dirc
	if not os.path.exists('out/png'):
		os.makedirs('out/png')
	if not os.path.exists('out/text'):
		os.makedirs('out/text')

	# Declaration variable and dictionary
	dict_sta = {}
	dict_event = {}
	win_snr = 2
	win_len = params["win_len"]
	snr = params["snr"]
	least_station = params["least_station"]
	station_list = params["station_list"]
	events_catalog = params["events_catalog"]
	#the length of amplitude
	namp_len = math.ceil(2*win_len*15)


	print("ok")
	print_c("Open inventory --> ")
	# Inventory declaration 
	inventory = "2019-10-21_ECGSNetworks_v0.7_KV-AF_SC3-OVG.xml"
	inv = read_inventory(inventory)
	print("ok")



	# Fill Dictionary
	print("Open station info and events catalog and write them in dictionary and dataframe:")
	input_x("click to continue...")
	station_info = pd.read_table(station_list, header=None)
	for i in range(len(station_info.values)):
		dict_sta[station_info.values[i][0]] = i
	event_info = pd.read_table(events_catalog, delimiter=r'\s+')
	# round the strattime to 2 digit to be able to compare with file starttime
	event_info['StartTime'] = event_info['StartTime'].apply(lambda x: re.match(r".*\...", x)[0])
	for i in range(0, len(event_info.values)):
		# -2 correspond to starttime event 
		dict_event[event_info['StartTime'][i]] =	[[] for i in range(len(station_info.values))]
	
	print_x("dict_event =")
	print_x(dict_event)	
	print_x("event_info =")
	print_x(event_info)
	print_x("dict_sta = ")
	print_x(dict_sta)
	input_x("click to continue...")



	# Create merged_event dictionary to keep only common event
	print("Create merged_event dictionary to keep only common event")
	merged_event = event_info
	keep_columns = "OriginTime , StartTime, EventID_x"
	selected_columns = [col.strip() for col in keep_columns.split(',')]

	# Filter merged_event catalog with station catalog one by one (keep only if OriginTime match for all stations)
	print("event catalog size  = ", len(merged_event))
	print("Evenet catalog will be filtered by common element with following station:")
	for name, param in dict_sta.items():
		station_CAT = glob.glob("PhD_data_class/CAT_2021/DAT/KV/"+name+"/CAT_IN*")
		for sta_cat in station_CAT:

			sta_event_info = pd.read_table(sta_cat, delimiter=r'\s+')
			# round the strattime to 2 digit to be able to compare with file starttime
			merged_event = pd.merge(merged_event, sta_event_info, on='OriginTime')
			# keep only interested columns
			merged_event = merged_event[selected_columns]
			print("--> {}: {} event remaining".format(name, len(merged_event)))

	# Ask user to limit the number of event to be calculated
	ev_num_max = input("Catalog common event finished, enter a value to limit number of event ineeded, otherwise press Enter!")
	stream_length = 0
	noise_length = -1
	while stream_length == 0:
		stream_length = int(input("Please enter the length of the stream [sec]:"))
	while noise_length < 0 or noise_length > stream_length:
		noise_length = int(input("Please enter the length of the noise [sec]:"))
	try:
		if int(ev_num_max) > 1:
			merged_event = merged_event.sample(n=int(ev_num_max), random_state=42)
	except:
		print("error in user data input, keep the entire catalog event")


	input("All your streams will be sliced on a length of {} sec with {} sec of noise before P, press enter to continue...".format(str(stream_length), str(noise_length)))



	print_x("merged_event = ")
	print_x(merged_event)
	input_x("click to continue...")



	print("")
	print("Loop through each station folder, then loop through eache event and open sequentially each corresponding mseed files found in catalog common.")
	print("Do the following for each station")
	print("  ->  Open stream, remove response and apply band filter")
	print("  ->  Extract starttime from stream")
	print("  ->  Extract start_time and origin_time from events catalog")
	print("  ->  Extract corresponding p_arr_man (which is pick manual by Jos in CAT_IN catalog of the corresponding station)")
	print("  ->  Cut the stream 3 sec before the pick and 17 sec after the pick")
	print("  ->  Calculate SNR value, fft_amp and amp_nmlz if noise is under threshold")
	print("  ->  Fill dict_event with these value")
	print("  ->  Display 2 method of plotting the sismo stream if requested")
	print("  ->  Plot all the results in the out_xxx_yyy foler (xxx = events number and yyy = station name)")
	input_x("click to continue...")


	# dict_event need to be reinitialise at each loop of station since we plot unique station,keep an original form
	# of the dict_event free from data station
	dict_event_original = dict_event

	for name, param in dict_sta.items():
		print("\r---------------------------------------")
		print("station  = ", name ,"  param = ", param)

		# Reset station data from dict_event
		dict_event = dict_event_original

		diff_stt_parr = []
		count_sta_event = 0
		for event_id in merged_event['EventID_x']:

			station_dir = glob.glob("PhD_data_class/CAT_2021/DAT/KV/"+name+"/"+str(event_id)+"/*")
			for sacname in station_dir :
				try:
					st = read(sacname)
					tr = read(sacname)[0]

				except:
					print(sacname, " is not a readable file for obspy !!!")
					continue
				t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
				if t[-1] < 10:
					continue

				# Manage counting on terminal to avoid undetermined lenght process
				count_sta_event += 1
				total_count = len(merged_event)
				sys.stdout.write('\r' + f"Progress: {count_sta_event}/{total_count}")
				sys.stdout.flush()

				# Correction and filtering of signals
				tr.remove_response(inventory=inv)
				tr.filter(type='bandpass', freqmin=1, freqmax=15)

				rate = tr.stats.sampling_rate
				datafull = tr.data
				# Data = datafull[100:-100]
				# Data = tr.data

				
				# read starttime from stream (file mseed)
				starttime = tr.stats.starttime
				# round to 2 digit after sec
				starttime = re.match(r".*\...", str(starttime))[0]
				# format starttime to be able able to compoare string with event_info
				starttime = starttime.replace("T", "-")

				# Read from merged dataframe origin and starttime
				origin_time = merged_event.loc[merged_event['EventID_x'] == int(event_id), 'OriginTime']
				start_time = merged_event.loc[merged_event['EventID_x'] == int(event_id), 'StartTime']

				# convert to datetime object 
				origin_time_ = datetime.strptime(origin_time.values[0], "%Y-%m-%d-%H:%M:%S.%f")
				start_time_ = datetime.strptime(start_time.values[0], "%Y-%m-%d-%H:%M:%S.%f")

				p_arr_man = picker_new(sacname)
				if not p_arr_man:
					continue
	
				# convert p_arr to datetime object
				p_arr_man_ = datetime.strptime(str(p_arr_man), "%S.%f")
				noise_time_  = origin_time_ - start_time_ + timedelta(seconds = float(p_arr_man))
				pick_abs_ = origin_time_ + timedelta(seconds = float(p_arr_man))

				noise_time = float(noise_time_.total_seconds())
				noise_time_pt = noise_time * rate

				# Cut the stream to have 3 seconds before the Pick and seventeen after
				cut_start_ = noise_time_  - timedelta(seconds = noise_length)
				cut_start = float(cut_start_.total_seconds())
				cut_start_pt = int(cut_start * rate)

				# Keep the 20 seconds of the cutted stream
				cut_end_pt = int((stream_length * rate) + cut_start_pt)
				Data = datafull[cut_start_pt:cut_end_pt]
				

				# To use to calculate noise but be careful that p_arr can be < 2 in several cases
				p_arr = noise_length

				# if p_arr > win_snr and p_arr < t[-1]-win_snr and quality > snr:
				if p_arr > win_snr and p_arr < t[-1]-win_snr:

					snr_value = SNR(Data, p_arr, rate, win_snr)
					if snr_value > snr:
						amp, freq, fi = fft_amp(Data, rate, p_arr, win_len)
						Amp_nmlz, freq_nmlz = amp_nmlz(freq, amp)
						name_event = starttime


						if (name_event in dict_event.keys()):
							dict_event[name_event][param].append(Amp_nmlz)
							dict_event[name_event][param].append(fi)
							# heavy but add by mjas to plot stream
							dict_event[name_event][param].append(Data)

				#--------------------------------------------------------
				########### Display plot (plot the event) ################
				
				if test_pick:
				
			

					# Extract x-axis values (time)
					print("tr.stats.delta = ", tr.stats.delta)


					# Define data ax1
					x_1 = np.arange(len(datafull))

					specific_time_1	 = x_1[0] + noise_time_pt
					y_1 = datafull
					supper_1 = np.ma.masked_where(x_1 < specific_time_1, x_1)
					slower_1 = np.ma.masked_where(x_1 > specific_time_1, x_1)
				
					# Define data ax1
					x_2 = np.arange(len(Data)) * tr.stats.delta
					specific_time_2	 = x_2[0] + noise_length
					y_2 = Data
					supper_2 = np.ma.masked_where(x_2 < specific_time_2, x_2)
					slower_2 = np.ma.masked_where(x_2 > specific_time_2, x_2)
				
					# Create a figure
					fig = plt.figure(figsize=(24, 14))

					plt.subplot(2, 1, 1)

					plt.plot(slower_2, y_2, supper_2, y_2)	
					plt.title(event_id)
					plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

					# Plot in obspy
					print(" ---- start plotting obspy -----")
					# Specify the desired time interval (start and end times)
					start_time_obspy_ = start_time_ + cut_start_
					cut_start_obspy = datetime.strftime(start_time_obspy_, "%Y-%m-%dT%H:%M:%S.%f")
					cut_end_ = start_time_obspy_ + timedelta(seconds = stream_length)
					cut_end_obspy = datetime.strftime(cut_end_, "%Y-%m-%dT%H:%M:%S.%f")

					pick_abs_obspy= datetime.strftime(pick_abs_, "%Y-%m-%dT%H:%M:%S.%f")


					start_time = UTCDateTime(cut_start_obspy)  # Replace with your desired start time
					end_time = UTCDateTime(cut_end_obspy)    # Replace with your desired end time
					pick_time = UTCDateTime(pick_abs_obspy) 

					# Filter the Stream to retain only data within the specified time interval
					stream_interval = st.slice(starttime=start_time, endtime=end_time)
					tr2 = stream_interval[0]
					tr2.remove_response(inventory=inv)
					tr2.filter(type='bandpass', freqmin=1, freqmax=15)


					print(st[0].stats.starttime)
					print(tr2.stats.starttime)

					# Extract timestamps (in seconds from the start time) from the sliced Stream
					timestamps = tr2.times()  # Assuming there is only one channel in the stream
					# create an array of timedelta
					timedelta_objects = [timedelta(seconds=x) for x in timestamps]

					print("timedelta_objects[0] = ", timedelta_objects[0])

					# extract strattime from obspy stream and convert the UTCDatetime to DateandTime
					starttime_obspy = tr2.stats.starttime
					starttime_ = UTCDateTime(starttime_obspy).datetime


					print("starttime_ = ", starttime_)
					# Create array of time axis for sliced stream with datetime object
					datetime_objects = [x + starttime_ for x in timedelta_objects ]

					print("datetime_objects[0] = ", datetime_objects[0])

					# Extract data from the sliced Stream
					data = tr2.data  # Assuming there is only one channel in the stream
					plt.subplot(2, 1, 2)

					plt.plot(datetime_objects, data)
					plt.axvline(x=pick_abs_, color='red', linestyle='--')

					# Customize the plot (optional)
					plt.title(event_id)
					plt.xlabel('Time')
					plt.ylabel('Amplitude')

					# Rotate and format the x-axis labels for better readability
					plt.xticks(rotation=20, ha='right')
					plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
					plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)



					# Create the quit button
					button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])  # Adjust the position and size as needed
					quit_button = Button(button_ax, 'Quit')

					# Add the quit button click event handler
					quit_button.on_clicked(quit_button_clicked)

					plt.show()


					########### Ending  Display plot (plot the event) ################
					#--------------------------------------------------------------------#

		###########################################################################
		print("")
		input_x("Display updated dict_event dict, continue...")
		print_c("dict_event = ")
		print_x(dict_event)
		print_c("dict_event length = ")
		print(len(dict_event))
		input_x("click to continue...")
		print("")
		print("remove dict function execution")
		input_x("click to continue...")
		dict_event = remove_dict(dict_event, dict_sta, namp_len)

		print_x("dict_event = ")
		print_x(dict_event)
		print_x("dict_event key = ")
		print_x(dict_event.keys())
		print_c("dict_event length = ")
		print(len(dict_event))
		input_x("click to continue...")
		# input("Datas are read, Click to continue to ....")
		
		print("Get Median Value")
		median_FI, median_amp, events_key, waveform = get_median_value(dict_event, namp_len, whether_plot)


		print("Calculate EM")
		median_distance = calculate_EM(median_amp)
		print("Plot Matrix")
		plot_matrix(median_distance)
		print("Plot dendrogram")
		plot_dendrogram(median_distance, params["n_cls"])
		#Hierarchical clustering
		print("AgglomerativeClustering")
		clust = AgglomerativeClustering(n_clusters=params["n_cls"],
										linkage='complete',
										affinity='precomputed').fit(median_distance)
	

		#dendrogram
		# print("clust_stats")
		clust_stats(clust, whether_plot)
		print("clust  = ", dir(clust))
		print("clust labels = ", clust.labels_)
		input("click to continue...")	
		# plot mean spectra
		print("plot mean spectra")
		mean_amp_container, sort_idx = freq_sort(clust.labels_, median_amp, params["n_cls"])

		len_idx = plot_mean_spectra(clust.labels_, median_amp, median_distance, mean_amp_container, sort_idx)
		#plot 100 spectra
		print("plot 100 spectra")


		print("median_amp  = ", median_amp)
		input("click to continue...")
		print("sort_idx  = ", sort_idx)
		print(len(waveform))
		print(len(events_key))
		print(len(median_amp))
		input("click to continue...")	

		plot_rep(clust, median_amp, sort_idx, waveform, events_key, noise_length, rate)
		# plot freq-evengy space
		print("freq-evengy space")
		freq_energy_distribution(mean_amp_container, freq, sort_idx, len_idx)
		#save files
		print("save new catalog")
		new_catalog(clust.labels_, event_info, events_key, median_FI, sort_idx)

		# Move results in final directory with station name
		station_folder = "out_{}_{}".format(len(dict_event), name)
		shutil.copytree('out', station_folder, dirs_exist_ok=True)










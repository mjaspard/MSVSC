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
from matplotlib.widgets import Button
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


####################################################
"""
This part is data processing.
"""

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
	return amp[(freq>=0) & (freq<15.0)], freq[(freq>=0) & (freq<15.0)], math.log10(A_upper/A_lower)


def picker(data, rate, nsta=10, nlta=100, uncer=0.5):

	# print("test")
	# print(data)
	# print(len(data))
	# plot_array(data)
	stalta = recursive_sta_lta(data, nsta, nlta)
	# print("test")
	# print(stalta)
	# print(len(stalta))
	# plot_array(stalta)

	n_max = np.argmax(stalta)
	# print("n_max = ",n_max)
	quality = stalta[n_max]
	# print("quality = ",quality)
	n_onset = int(uncer * rate)
	# print("n_onset = ",n_onset)
	# add
	diff_max = np.diff(stalta[n_max-n_onset:n_max+n_onset])
	# print("diff_max = ", diff_max)
	# print(len(diff_max))
	# end
	n_diff_max = np.argmax(np.diff(stalta[n_max-n_onset:n_max+n_onset]))
	# print("n_diff_max = ", n_diff_max)
	n_pick = n_max - n_onset + n_diff_max
	# print("n_pick = ", n_pick)
	arr = n_pick / rate
	# print("arr = ", arr)
	# input("wait...")
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
		# print(num_event, " not station CAT_IN")
		return False
	# print("station = ", dirname2)
	# print("EventID =" , int(num_event))
	# print("arr_P.values[0] =" , arr_P.values[0])
	# input("wait...")
	return arr_P.values[0]

def plot_array(array):
    plt.plot(array)
    plt.show()


	# input("wait...")

################################################################################################
def remove_dict(dict_event, dict_sta, namp_len):
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
	median_FI = []
	median_amp = []
	events_key = []
	for key, value1 in dict_event.items():
		tempo_FI = []
		tempo_amp = []
		events_key.append(key)
		for para in dict_sta.values():
			if (value1[para] != []):
				if (len(value1[para][0]) == namp_len):
					tempo_FI.append(value1[para][1])
					tempo_amp.append(value1[para][0])
		if len(tempo_FI)%2 == 1:
			median_as = tempo_FI.index(np.median(tempo_FI))
			if isinstance(median_as, int):
				median_FI.append(tempo_FI[median_as])
				median_amp.append(tempo_amp[median_as])
			else:
				median_as = median_as[0]
				median_FI.append(tempo_FI[median_as])
				median_amp.append(tempo_amp[median_as])
		else:
			tempo_FI.pop(0)
			median_as = tempo_FI.index(np.median(tempo_FI))
			if isinstance(median_as, int):
				median_FI.append(tempo_FI[median_as])
				median_amp.append(tempo_amp[median_as])
			else:
				median_as = median_as[0]
				median_FI.append(tempo_FI[median_as])
				median_amp.append(tempo_amp[median_as])
	bins_ = np.linspace(-1.5, 1, 30)
	plt.hist(median_FI, bins=bins_, edgecolor='k')
	plt.savefig('out_jos/png/median_FI.png', format = 'png')
	pickle.dump(np.asarray(events_key), open('out_jos/text/events_key.pkl', 'wb'))
	pickle.dump(np.asarray(median_amp), open('out_jos/text/median_amp.pkl', 'wb'))
	if whether_plot:
		plt.show()
	plt.close()
	return median_FI, median_amp, events_key


def calculate_EM(median_amp):
	sum_amp = np.sum(median_amp[0]) * np.sqrt(2)
	median_distance = np.zeros((len(median_amp), len(median_amp)))
	for i in range(len(median_amp)):
		for j in range(i+1,len(median_amp)):
			median_distance[i, j] = np.sqrt(np.sum((median_amp[i][:]-median_amp[j][:])**2))/sum_amp
			median_distance[j, i] = median_distance[i, j]
			
	pickle.dump(np.asarray(median_distance), open('out_jos/text/median_distance.pkl', 'wb'))
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
	title = 'out_jos/png/hist.png'
	plt.savefig(title, format='png')
	if whether_plot:
		plt.show()
	plt.close()


def new_catalog(labels, event_info, events_key, median_FI, sort_idx):
	k = 1
	print("new_catalog step 1")
	new_events_catalog = []
	event_info.values[0][0] = str(event_info.values[0][0])  \
							  + ' FI' + ' CLUSTER'
	for i in range(len(events_key)):
		print("1 - ", i)
		for j in range(k, len(event_info.values)):
			if events_key[i] == str(event_info.values[j][0]).split(' ')[0]:
				event_info.values[j][0] = str(event_info.values[j][0]) + ' ' \
					+ str(round(median_FI[i], 2)) + ' ' + str(np.where(sort_idx == labels[i])[0][0])
				new_events_catalog.append(str(event_info.values[j][0]))
				k = j
				break
	print("new_catalog step 2")
	for i in range(len(sort_idx)):
		print("2 - ", i)
		idx_cls = np.where(labels == sort_idx[i])[0]
		with open('./out_jos/text/cluster'+str(i)+'.dat', 'w') as f:
			for j in range(len(idx_cls)):
				f.write(new_events_catalog[idx_cls[j]] + '\n')

	event_info.to_csv('out_jos/text/new_catalog', index=False, header=None)
				 

def plot_rep(clust, amp, sort_idx):
	labels = np.unique(clust.labels_)
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
		title = 'out_jos/png/cls'+str(label)+'.pdf'
		plt.savefig(title, format='pdf')
		plt.close()

#turn it to two function. 1.get the sirt_idx; 2. plot the median spectra
def freq_sort(labels, amp, n_cls):
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
	len_idx = []
	amp = np.array(amp)
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
		plt.plot(amp[idx_cls][imin], linewidth=2, c='#333333')
		plt.plot(mean_amp_container[sort_idx[i]], linewidth=2, c='#FF3333')
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
	plt.savefig('out_jos/png/mean_spectra.pdf', format='pdf')
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
	plt.savefig('out_jos/png/fre_energy.pdf', format='pdf')
	if whether_plot:
		plt.show()
	plt.close()	  

	np.savetxt("out_jos/text/peak_amp_size", list(zip(peak_freq, 1./max_amp, 0.1*np.log10(np.array(len_idx)), np.arange(20))))



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
	plt.savefig('out_jos/png/med_dis.pdf', format = 'pdf')
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
	plt.savefig('out_jos/png/dendrogram.pdf', format='pdf')
	if whether_plot:
		plt.show()
	plt.close()
###################################################################


if __name__ == '__main__':
	#get para from config_json
	parser = ap.ArgumentParser(
		prog='Muti-station_volcanic_signals_classification.py',
		description='classified volcano signals')
	parser.add_argument('config_json_jos')
	parser.add_argument(
		'-P',
		default=False,
		action='store_true',
		help='Plot output')
	args = parser.parse_args()
	whether_plot = args.P

	with open(args.config_json_jos, "r") as f:
		params = json.load(f)

	# make dirc
	if not os.path.exists('out_jos/png'):
		os.makedirs('out_jos/png')
	if not os.path.exists('out_jos/text'):
		os.makedirs('out_jos/text')

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


	# Inventory declaration 
	inventory = "2019-10-21_ECGSNetworks_v0.7_KV-AF_SC3-OVG.xml"
	inv = read_inventory(inventory)

	# Test the picker by showing each stream
	test_pick = input("Check picking ? [y/n]")
	if re.match("[y/Y]", test_pick):
		test_pick = True
	else:
		test_pick = False


	# Fill Dictionary
	station_info = pd.read_table(station_list, header=None)
	for i in range(len(station_info.values)):
		dict_sta[station_info.values[i][0]] = i
	event_info = pd.read_table(events_catalog, delimiter=r'\s+')
	# round the strattime to 2 digit to be able to compare with file starttime
	event_info['StartTime'] = event_info['StartTime'].apply(lambda x: re.match(r".*\...", x)[0])
	for i in range(0, len(event_info.values)):
		# -2 correspond to starttime event 
		dict_event[event_info['StartTime'][i]] =	[[] for i in range(len(station_info.values))]
	

	# Create merged_event dictionary to keep only common event
	merged_event = event_info
	keep_columns = "OriginTime , StartTime, EventID_x"
	selected_columns = [col.strip() for col in keep_columns.split(',')]

	# Filter merged_event catalog with station catalog one by one (keep only if OriginTime match for all stations)
	print("event catalog size  = ", len(merged_event))
	print("Evenet catalog will be filtered by common element with following station:")
	for name, param in dict_sta.items():
		station_CAT = glob.glob("PhD_data_class/CAT_2019/DAT/KV/"+name+"/CAT_IN*")
		for sta_cat in station_CAT:

			sta_event_info = pd.read_table(sta_cat, delimiter=r'\s+')
			# round the strattime to 2 digit to be able to compare with file starttime
			merged_event = pd.merge(merged_event, sta_event_info, on='OriginTime')
			# keep only interested columns
			merged_event = merged_event[selected_columns]
			print("--> {}: {} event remaining".format(name, len(merged_event)))

	# Ask user to limit the number of event to be calculated
	ev_num_max = input("Catalog common event finished, enter a value to limit number of event ineeded, otherwise press Enter!")
	try:
		if int(ev_num_max) > 1:
			merged_event = merged_event.sample(n=100, random_state=42)
	except:
		print("error in user data input, keep the entire catalog event")

	print(merged_event)
	input("wait...")




	for name, param in dict_sta.items():
		
		print("")
		print("---------------------------------------")
		print("station  = ", name ,"  param = ", param)

		count_sta_event = 0
		for event_id in merged_event['EventID_x']:

			station_dir = glob.glob("PhD_data_class/CAT_2019/DAT/KV/"+name+"/"+str(event_id)+"/*")
			for sacname in station_dir :
				try:
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
				Data = datafull[100:-100]
				# Data = tr.data

				
				# add for jos
				starttime = tr.stats.starttime
				# round to 2 digit after sec
				starttime = re.match(r".*\...", str(starttime))[0]
				# format starttime to be able able to compoare string with event_info
				starttime = starttime.replace("T", "-")

				# end
				
				# Original picker function
				# p_arr, quality = picker(Data, rate)
				# print("p_arr (original) = ", p_arr)
				# print("quality = ", quality)			

				p_arr_man = picker_new(sacname)
				if not p_arr_man:
					continue
				p_arr_man = p_arr_man - 2 # !!! because Data = datafull[100:-100] --> 
				# New picker functiom
				p_arr = p_arr_man

				# print("p_arr (jos) = ", p_arr_man)

				# input("wait...")


				#--------------------------------------------------------
				########### Display plot (plot the event) ################
				
				if test_pick:
				
				
					# Extract x-axis values (time)
					print("tr.stats.delta = ", tr.stats.delta)
					x = np.arange(len(Data)) * tr.stats.delta

					# Define the specific time from the beginning of the stream
					specific_time = x[0] + p_arr  

					# Extract y-axis values (amplitude)
					y = Data

					supper = np.ma.masked_where(x <= specific_time, x)
					slower = np.ma.masked_where(x >= specific_time, x)
				
					# print(x)
					# print(supper)
					# print(slower)			
				
					# Create a figure
					fig, ax = plt.subplots()

					ax.get_figure().set_figwidth(20)   # Width: 8 inches
					ax.get_figure().set_figheight(8)  # Height: 6 inches

				
					ax.plot(slower, y, supper, y)

					# add vetical ine where jos picked manually the event
					ax.axvline(x=p_arr_man, color='red', linestyle='--')
					ax.text(p_arr_man, max(y), 'Jos pick manual', color='red', fontsize=10)

					# Create the quit button
					button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])  # Adjust the position and size as needed
					quit_button = Button(button_ax, 'Quit')

					# Add the quit button click event handler
					quit_button.on_clicked(quit_button_clicked)
				
					# Show the plot
					plt.show()

				######### finished ##################
				#--------------------------------------------------------
				


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

###########################################################################

	print("")
	print("remove dict function start")
	dict_event = remove_dict(dict_event, dict_sta, namp_len)
	# print("dict_event key = ")
	# print(dict_event.keys())
	print("dict_event = ")
	print(dict_event)
	
	input("Datas are read, Click to continue to ....")
	
	print("Get Median Value")
	median_FI, median_amp, events_key = get_median_value(dict_event, namp_len, whether_plot)
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
	print("clust_stats")
	clust_stats(clust, whether_plot)
	# plot mean spectra
	print("plot mean spectra")
	mean_amp_container, sort_idx = freq_sort(clust.labels_, median_amp, params["n_cls"])
	len_idx = plot_mean_spectra(clust.labels_, median_amp, median_distance, mean_amp_container, sort_idx)
	#plot 100 spectra
	print("plot 100 spectra")
	plot_rep(clust, median_amp, sort_idx)
	# plot freq-evengy space
	print("freq-evengy space")
	freq_energy_distribution(mean_amp_container, freq, sort_idx, len_idx)
	#save files
	print("save file")
	new_catalog(clust.labels_, event_info, events_key, median_FI, sort_idx)












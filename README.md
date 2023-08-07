This repo is based on following repo: https://github.com/xcui1997/Multi-station-volcanic-seismicity-classification.git


The main code has been adapted to a specific events catalog, some fetures has been added and other removed.


1. The clustering is here adapted to work with only one station. We choose that way of working to avoid that the programs had to choose the signal from 1 station to represent an event, even if signals are available for others stations. Like this, all events from ecah station is considered.

2. We added following arguments options:
	-v 			--> Verbose mode with sequential run of the script to verify each step.
	--strem 	--> Display the Pick point on the stream to verify pick position
	-P 			--> Display results plot (already in the original version)


3. The pick of Primary is not anymore calculated but read from catalog of stations. (Already manually picked before)

4. The results of the scripts are written in output folder with the name 'out_xxx_yyy' (xxx = events number and yyy = station name)

5. Hard coded line are the following in the scripts:
	- 556 - catalog name file (full path if not in working directory)
	- 595 - Station event Catalog position from working dir based on the name of the station
	- 660 - Station event Catalog position from working dir based on the event id.



Here under brief description of code in main process:


- Write catalog events (all events) in a dataframe "event_info"
- Write list of station in a dataframe "station_info"
- Create a dictionary "dict_event" where all calculated infos will be written
- Create a dictionary "merged_event" which is a copy of event_info but ONLY with events that are in ALL stations (based on origin time)
- Users can limit the number of events to be calculated, and reduced the stream length to reduce calculation time

- Then, loop through each station and do the following:
	->  Open stream, remove response and apply band filter
	->  Extract starttime from stream
	->  Extract start_time and origin_time from events catalog
	->  Extract corresponding p_arr_man (which is pick manual in CAT_IN catalog of the corresponding station)
	->  Cut the stream x sec before the pick and x sec after the pick
	->  Calculate SNR value, fft_amp and amp_nmlz if noise is under threshold
	->  Fill dict_event with these value
	->  Display 2 method of plotting the sismo stream if requested
	->  Calculate all the stuff the same way as in the original version of the repo (cluster...)
	->  Plot all the results in the out_xxx_yyy foler (xxx = events number and yyy = station name)



To run the script, the mandatory argument must be named "config_json" and must be a json file with following parameters

{
    "data_dir": "/home/xincui",
    "snr": 2,
    "win_len": 1.28,
    "station_list": "station_list",
    "events_catalog": "CAT_IN_20210607_165737.txt",
    "least_station": 1,
    "n_cls": 8
}



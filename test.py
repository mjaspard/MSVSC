import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import numpy as np
from obspy import read
from obspy import UTCDateTime

# Load your seismic data file into an ObsPy Stream object
file = "/Users/maxime/Project/jos_msvsc/Multi-station-volcanic-seismicity-classification/PhD_data_class/CAT_2019/DAT/KV/BULE/201900002/KV.BULE..BHZ.D.2019.001"
stream = read(file)

# Specify the desired time interval (start and end times) in absolute time
start_time = UTCDateTime("2019-01-01T01:27:00")  # Replace with your desired start time
end_time = UTCDateTime("2019-01-01T01:29:00")    # Replace with your desired end time

# Filter the Stream to retain only data within the specified time interval
stream_interval = stream.slice(starttime=start_time, endtime=end_time)

# Extract timestamps (in seconds from the start time) from the sliced Stream
timestamps = stream_interval[0].times()  # Assuming there is only one channel in the stream


print(timestamps)
print(type(timestamps))
print(type(timestamps[0]))


# create an array of timedelta
timedelta_objects = [timedelta(seconds=x) for x in timestamps]

print(timedelta_objects)
print(type(timedelta_objects))
print(type(timedelta_objects[0]))




starttime = stream_interval[0].stats.starttime
starttime = UTCDateTime(starttime).datetime


# Convert timestamps to datetime objects
# datetime_objects = [UTCDateTime(ts).datetime for ts in timestamps]

# print(datetime_objects)



input()


datetime_objects = [x + starttime for x in timedelta_objects ]



# input("wait...")

# Extract data from the sliced Stream
data = stream_interval[0].data  # Assuming there is only one channel in the stream

# Plot the data with the datetime axis using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, data)

# Customize the plot (optional)
plt.title('Seismic Data with Datetime on X-axis')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Rotate and format the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))

# Show the plot
plt.tight_layout()
plt.show()
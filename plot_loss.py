"""
Quick and dirty script to plot the loss function along with epsilon

usage :
python plot-loss.py <logfilenumber> <smooth_factor>

where <logfilenumber> is a mandatory argument
and <smooth_factor> is default to 15 if not specified
"""


from matplotlib import pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print('Usage: python plot-loss.py <logfilenumber>')
    exit(1)
if len(sys.argv) > 2:
    smooth_factor = int(sys.argv[2])
else:
    smooth_factor = 15
if len(sys.argv) > 3:
    print('Dropping first ' + sys.argv[3] + ' values')
    drop = int(sys.argv[3])
else:
    drop = 0

logfile = os.path.join("logs", "log_" + sys.argv[1] + ".log")


extract_data = False  # Flag to indicate when to start extracting lines
lines = []  # List to store the extracted lines
with open(logfile, "r") as file:
    for line in file:
        if line.strip() == "loss,epsilon,perf":
            # Found the closing brace, start extracting lines
            extract_data = True
        elif extract_data:
            # Extract the lines below the closing brace
            lines.append(line.strip())


loss = [float(line.split(',')[0]) for line in lines]
epsilon = [float(line.split(',')[1]) for line in lines]
perf = [(float(line.split(',')[2])+1)/2 for line in lines]


# Now that we have the loss, plot a smooth version of it
# taking the mean of smooth_factor consecutive episodes
smooth_loss = []
smooth_perf = []

# Compute the smoothed values
for i in range(smooth_factor, len(loss)):
    smooth_value = sum(loss[i - smooth_factor:i]) / smooth_factor
    smooth_loss.append(smooth_value)
    perf_value = sum(perf[i - smooth_factor:i]) / smooth_factor
    smooth_perf.append(perf_value)

# Append the first values without smoothing
smooth_loss[:0] = loss[:smooth_factor]
smooth_loss = smooth_loss[drop:]
smooth_perf[:0] = perf[:smooth_factor]
smooth_perf = smooth_perf[drop:]

epsilon = epsilon[drop:]

X = range(len(smooth_loss))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)


ax1.plot(X, smooth_loss)
ax1.set_title('Loss function (smoothed)')


ax2.plot(epsilon)
ax2.set_title('Epsilon')

ax3.plot(smooth_perf)
ax3.set_title('Performance (smoothed)')

plt.show()

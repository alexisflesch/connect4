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

logfile = os.path.join("logs", "log_" + sys.argv[1] + ".txt")

with open(logfile, 'r') as f:
    lines = f.readlines()
    loss = [float(line.split(',')[0]) for line in lines]
    epsilon = [float(line.split(',')[1]) for line in lines]


# Now that we have the loss, plot a smooth version of it
# taking the mean of smooth_factor consecutive episodes
smooth_loss = []

# Compute the smoothed values
for i in range(smooth_factor, len(loss)):
    smooth_value = sum(loss[i - smooth_factor:i]) / smooth_factor
    smooth_loss.append(smooth_value)

# Append the first values without smoothing
smooth_loss[:0] = loss[:smooth_factor]

X = range(len(smooth_loss))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


ax1.plot(X, smooth_loss)
ax1.set_title('Loss function (smoothed)')


ax2.plot(epsilon)
ax2.set_title('Epsilon')

plt.show()

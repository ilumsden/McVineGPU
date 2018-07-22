import numpy as np
from matplotlib import pyplot as plt

x = np.array([1e6, 3e6, 6e6, 1e7, 3e7, 6e7, 1e8])
cputimes = np.array([50.0, 147.0, 293.0, 491.0, 1473.0, 3005.0, 4977.0])
gputimes = np.array([3.72244, 4.24126, 4.71751, 5.69286, 8.8244, 14.7054, 21.627])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, gputimes, ".-", color="#74B71B")
ax1.set_xlabel("Number of Neutrons")
ax1.set_ylabel("McVineGPU Execution Time (s)", color="#74B71B")
ax1.set_ylim(0, 25)
ax1.tick_params("y", colors="#74B71B")
#ax1.set_xscale("log")

ax2.plot(x, cputimes, ".-", color="#0000FF")
ax2.set_ylabel("MCViNE Execution Time (s)", color="#0000FF")
#ax2.set_ylim(0, 5000)
ax2.tick_params("y", colors="#0000FF")

fig.suptitle("Timing Comparison of MCViNE vs McVineGPU", y=0.95)

#A = np.vstack([x, np.ones(len(x))]).T
#m, c = np.linalg.lstsq(A, gputimes)[0]

#plt.plot(x, cputimes, ".-", color="#0000FF", label="MCViNE")
#plt.plot(x, gputimes, ".-", color="#74B71B", label="McVineGPU")
#plt.plot(x, m*x+c, "-", color="#74B71B", label="McVineGPU")

#plt.legend()

#plt.xscale("log")
#plt.yscale("log")

#plt.title("Timing Comparison of MCViNE vs McVineGPU")

#plt.xlabel("Number of Neutrons")
#plt.ylabel("Execution Time (s)")

#plt.grid()

plt.show()

import matplotlib.pyplot as plt
import json
import sys
import numpy as np

with open(sys.argv[1], 'rb') as f:
	x, y = zip(*json.load(f))

def bound(v):
	v = max(0.0, v)
	return min(1.0, v)

fig, ax = plt.subplots()
y = [bound(v) for v in y]
x = [bound(v) for v in x]
ax.scatter(x, y)

lims = [
	0, 1
]
fit = np.poly1d(np.polyfit(x, y, 1))
ax.plot(lims, lims)
ax.plot(x, fit(x))
plt.show()

# comment the above stuff (aside from imports) and comment the below stuff for graph of ackermann 
# with open('normalized_data/ackermann_v2.json', 'rb') as f:
# 	original = json.load(f)

# data = []
# for key,value in sorted(original.items()):
# 	data += value

# plt.plot(data, np.arange(len(data)))
# plt.show()

# note - series of 3 scans
import numpy as np
import json
import matplotlib.pyplot as plt 
import sys

with open(sys.argv[1], 'rb') as f:
	data = json.load(f)

plt.plot(np.arange(len(data)), data)
plt.ylim((0, 1))
plt.show()
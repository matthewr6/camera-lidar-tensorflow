import sys
import numpy as np
import random
import math
import json

import matplotlib.pyplot as plt

def open_file(f):
    with open('{}.json'.format(f), 'rb') as out:
        return json.load(out)

ackermann = open_file('normalized_data/ackermann')

data = []
for time in ackermann:
		data += ackermann[time]

y = np.arange(len(data))
x = data

plt.hist(x, y)
plt.show()
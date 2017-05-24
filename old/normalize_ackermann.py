import sys
import numpy as np
import random
import math
import json

laser_only = False
camera_only = True
verbose_batch = False
very_verbose_batch = False

def open_file(f):
    with open('{}.json'.format(f), 'rb') as out:
        return json.load(out)

print 'opening files'
ackermann = open_file('dict_data/ackermann')
print 'files opened'

# 10, 100, 1000
factor = 100.0
for time in ackermann:
	for idx, item in enumerate(ackermann[time]):
		ackermann[time][idx] = item * factor

def save_file(name, data):
    with open('{}.json'.format(name), 'wb') as f:
        json.dump(data, f)

# try scaling exponentially
save_file('normalized_data/ackermann', ackermann)

# differentiate between small differences - cdf/pdf/normal distributions
# more likely on output? "spreading out output data"
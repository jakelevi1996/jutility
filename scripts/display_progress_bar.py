import time
import numpy as np
from jutility import util

rng = np.random.default_rng(0)
t = 7
num_steps = 10000
for i in util.progress(range(num_steps)):
    time.sleep(rng.uniform(0, 2 * t / num_steps))

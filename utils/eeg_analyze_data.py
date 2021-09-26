import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque

style.use("ggplot")

fps_counter = deque(maxlen=100)

FPS = 105
HM_SECONDS_SLICE = 10

data = np.load("./data/seq-3000.npy")
print(len(data))
print(FPS*HM_SECONDS_SLICE)

for i in range(FPS*HM_SECONDS_SLICE, len(data)):
	new_data = data[i-FPS*HM_SECONDS_SLICE: i]
	c2 = new_data[:, 2]

	GRAPH = c2
	print(c2)
	time.sleep(1/FPS)

	plt.plot(c2)
	plt.show()
	break

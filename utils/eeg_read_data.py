
import cv2
import time
import numpy as np
from collections import deque
from pyOpenBCI import OpenBCIGanglion

num_of_samples = 3000
last_print = time.time()
fps_counter = deque(maxlen=50)
sequence = np.zeros((num_of_samples, 4)) # 100 - num of samples
counter = 0

def print_raw(sample):
	global last_print
	global sequence
	global counter
	
	sequence = np.roll(sequence, 1, 0)
	sequence[0, ...] = sample.channels_data

	fps_counter.append(time.time() - last_print)
	last_print = time.time()
	print(f'FPS: {1/(sum(fps_counter)/len(fps_counter)):.2f}, {len(sequence)}, ... {counter}')
	
	counter += 1
	if counter == num_of_samples:
		np.save(f"./data/sa/hear_sesnse_seq-3000_1.npy", sequence)
		print('file saved!')
		exit(0)

board = OpenBCIGanglion()
board.start_stream(print_raw)

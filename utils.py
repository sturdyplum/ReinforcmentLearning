import numpy as np

def preprocess(data):
    # Other games:
    reduced = data[0:-1:2, 0:-1:2]
    grayscale = np.sum(reduced, axis=2)
    return np.expand_dims(grayscale, axis=2)/255.0

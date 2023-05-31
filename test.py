"""HelloWorld"""

import numpy as np

def generate_x_fnc(N) -> np.ndarray:
    X_1 = np.random.normal(loc = 1, size = N)
    X_2 = np.random.normal(loc = 1, size = N) #loc (mean) should be higher because we want X to be distributed around positive numbers
    
    return X_1, X_2
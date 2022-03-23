
import numpy as np
from jax import random

mapping_size = 16
rand_key = random.PRNGKey(0)
B_gauss = random.normal(rand_key, (mapping_size, 2))
np.save("B_gauss.npy", B_gauss)


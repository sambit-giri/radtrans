import numpy as np
from glob import glob 
import os
import pyc2ray1d

out = pyc2ray1d.grid_ini_fn(0.1,1e23)
print(out)
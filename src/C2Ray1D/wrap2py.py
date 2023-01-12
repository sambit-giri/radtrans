import numpy as np 
import os, sys 

os.system('bash makebashfile.sh')

sys_path = np.array(sys.path)[np.array(['radtrans' in ff for ff in sys.path])][0]
sys_path = sys_path.split('radtrans')[0]
print(sys_path)

os.system('cp pyccray*.so {}'.format(sys_path))

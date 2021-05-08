'''
Created on 16 Apr 2020
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

setup(name='sim21cm',
      version='0.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@ics.uzh.ch',
      package_dir = {'sim21cm' : 'src'},
      packages=['sim21cm'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy', 'tools21cm@git+https://github.com/sambit-giri/tools21cm.git', 'joblib', 'tqdm'],
      #include_package_data=True,
)

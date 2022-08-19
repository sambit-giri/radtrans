'''
Created on 8 May 2021
@author: Sambit Giri, Timothée Schaeffer
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

setup(name='radtrans',
      version='0.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@ics.uzh.ch',
      package_dir = {'radtrans' : 'src'},
      packages=['radtrans'],
      package_data={'radtrans':['input_data/*.dat','files/*.dat']},
      install_requires=['numpy','scipy', 'joblib', 'tqdm', 'astropy', 
                        'scikit-image', 'scikit-optimize'],
      #include_package_data=True,
)

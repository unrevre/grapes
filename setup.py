from distutils.core import setup, Extension
from os import environ

venv_path = environ['VIRTUAL_ENV']
numpy_dir = '/lib/python3.9/site-packages/numpy/core/include/numpy/'

module = Extension('wine',
                   include_dirs = [venv_path + numpy_dir],
                   sources = ['grapes/lib/wine.c'])

setup(name = 'grapes',
      version = '0.1.0',
      description = 'grapes!',
      ext_modules = [module])

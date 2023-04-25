from distutils.core import setup, Extension

module = Extension('wine',
                   sources = ['grapes/lib/wine.c'])

setup(name = 'grapes',
      version = '0.1.0',
      description = 'grapes!',
      ext_modules = [module])

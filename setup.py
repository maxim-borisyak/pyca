"""
  Py Cellular Automata.
"""

from setuptools import setup, find_packages, Extension

from codecs import open
import os
import os.path as osp
import numpy as np

def get_includes():
  env = os.environ

  includes = []

  for k in ['CPATH', 'C_INCLUDE_PATH', 'INCLUDE_PATH']:
    if k in env:
      includes.append(env[k])

  return includes

def get_library_dirs():
  env = os.environ

  libs = []

  for k in ['LD_LIBRARY_PATH']:
    if k in env:
      libs.append(env[k])

  return libs

from Cython.Build import cythonize

here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

extra_compile_args=['-std=c++11', '-O3', '-D_GLIBCXX_USE_CXX11_ABI=0']

extensions = [
  Extension(
    'pyca.pyca_ops', ['pyca/pyca_ops.pyx'],
    libraries=['stdc++'],
    include_dirs=[np.get_include()] + get_includes(),
    library_dirs=get_library_dirs(),
    language='c++',
    extra_compile_args=extra_compile_args
  )
]

setup(
  name='pyca',

  version='0.1.1',

  description="""Python Cellular Automata""",

  long_description = long_description,

  url='https://github.com/maxim-borisyak/pyca',

  author='Maxim Borisyak',
  author_email='mborisyak at yandex-team dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at yandex-team dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
  ],

  keywords='Cellular Automata',

  packages=find_packages(exclude=['contrib', 'examples', 'docs', 'tests']),

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'numpy',
    'cython'
  ],

  include_package_data=True,

  package_data = {
  },

  ext_modules = cythonize(extensions, gdb_debug=True),
)

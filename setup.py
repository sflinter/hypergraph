from setuptools import setup
from setuptools import find_packages

long_description = '''
Hypergraph is a python library for implementing graphs and networks (eg. neural networks) where meta-heuristic
algorithms (eg. genetic algorithms) can alter the structure with the purpose of finding the optimal
arrangement for a specific task.

Hypergraph is compatible with Python 3.6
and is distributed under the Apache 2.0 license.
'''

setup(name='Hypergraph',
      version='0.0.1',
      description='Graph based meta-heuristic algorithms',
      long_description=long_description,
      author='Nicola Mariella',
      author_email='ejpiminus1eq0@gmail.com',
      url='https://github.com/aljabr0/hypergraph',
      download_url='https://github.com/aljabr0/hypegraph/tarball/master',
      license='Apache 2.0',
      install_requires=['numpy>=1.9.1',
                        'pandas>=0.23.3',
                        'msgpack'],
      extras_require={
          'hp': ['hyperopt']
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())

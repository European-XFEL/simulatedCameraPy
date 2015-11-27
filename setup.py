#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='simulatedCameraPy',
      version='',
      author='',
      author_email='',
      description='',
      long_description='',
      url='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      entry_points={
          'karabo.python_device.api_1': [
              'SimulatedCameraPy = simulatedCameraPy.SimulatedCameraPy:SimulatedCameraPy',
          ],
      },
      package_data={},
      requires=[],
      )


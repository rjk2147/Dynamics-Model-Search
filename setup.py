from setuptools import setup

setup(
   name='dynamics_model_search',
   version='1.0',
   description='Dynamics Model Search',
   author='Robert Kwiatkowski',
   author_email='robert.kwiatkowski@columbia.edu',
   packages=['dynamics_model_search'],  #same as name
   install_requires=['torch', 'numpy', 'pybullet', 'matplotlib']
)

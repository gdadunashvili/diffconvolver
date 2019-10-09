from setuptools import setup

setup(
   name='diffconvolver',
   version='0.1',
   description='Solve pde systems using FDM',
   author='George Dadunashvili',
   author_email='g.dadunashvili@mailbox.org',
   packages=['diffconvolver'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)
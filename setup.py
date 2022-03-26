from importlib_metadata import version
from setuptools import setup
import json
name ='diffconvolver'
with open("VERSION.json", 'r') as v:
   version=json.loads(v.read())["VERSION"]
setup(
   name=name,
   version='0.1',
   description='Solve pde systems using FDM',
   author='George Dadunashvili',
   author_email='g.dadunashvili@mailbox.org',
   packages=['diffconvolver'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)
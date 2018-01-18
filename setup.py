from __future__ import print_function

from os import environ
from os.path import realpath
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_call
from sys import executable as python_binary_path, platform, version_info
import tarfile

if version_info.major == 2:
  from urllib import urlretrieve
else:
  from urllib.request import urlretrieve

pyglrm_version = '0.1.0'



#address_pyjulia = 'git+https://github.com/JuliaPy/pyjulia'
msg_configurejulia = 'Ensuring Julia has required packages installed...'
msg_juliaerror = 'Failed to run Julia.  Please ensure that Julia is properly installed on this\nsystem and then rerun the installer.'
msg_needjulia = 'Julia not found on this system.'
msg_needjulia_auto = 'We will try to automatically install and configure Julia for use with this\npackage.'
msg_install_failed = 'This package requies Julia and the installer does not know how to install Julia\non this system. Please install Julia manually and then rerun this installation.'
msg_postinstall = 'Setup has installed Julia.  The Julia directory can be found in the environment\nvariable PATH after the terminal is restarted.  For use immediately, run\nexport PATH=$PATH:$HOME/.julia/julia-903644385b/bin'



def installjulia():
  try:  # Set error names by Python version.
    RunTimeError
  except NameError:
    RunTimeError = RuntimeError
    
  print(msg_needjulia_auto)
  

  if platform.startswith('linux'):
    check_call([python_binary_path, 'setup-linux.py']) 
  else:
    raise RunTimeError(msg_install_failed)
  
  print(msg_postinstall)


def configurejulia():
  try:  # Set error names by Python version.
    FileNotFoundError
  except NameError:
    FileNotFoundError = OSError
  try:
    RunTimeError
  except NameError:
    RunTimeError = RuntimeError
  
  print(msg_configurejulia)
  
  try:
    check_call(['julia', 'setup.jl', python_binary_path, realpath('setup.sh')])
    return
  except FileNotFoundError:
    print(msg_needjulia)
  
  installjulia()
  
  try:
    check_call(['julia', 'setup.jl', python_binary_path, realpath('setup.sh')])
    return
  except FileNotFoundError:
    raise RunTimeError(msg_juliaerror)
    

class CustomInstall(install):
  def run(self):
    configurejulia()
    install.run(self)
    

setup(
  name='pyglrm',
  version=pyglrm_version,
  description='LowRankModels.py is a python package for modeling and fitting generalized low rank models (GLRMs).',
  author='Matthew Zalesak, Anya Chopra & Madeline Udell',
  author_email='mdz32@cornell.edu',
  url='https://github.coecis.cornell.edu/mdz32/pyglrm',
  include_package_data=True,
  install_requires=["julia", "numpy"],
  setup_requires=["julia"],
  package_data={'' : ['README.md']},
  license='MIT',
  packages=['pyglrm'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Low Rank Models',
    'Topic :: Software Development :: Libraries :: Python Modules'],
  cmdclass={'install': CustomInstall}
  )


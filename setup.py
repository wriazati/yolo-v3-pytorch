#! /usr/bin/env python

from os.path import dirname, realpath, join
from setuptools import setup, find_packages


####
# Basic metadata.
####

project_name = 'yolo-v3-pytorch'
package_name = project_name
repo_name    = project_name
src_subdir   = 'src'
description  = 'You Only Look Once in PyTorch'
url          = 'https://github.com/wriazati/' + repo_name
author       = 'wriazati'
author_email = author + '@yahoo.com'


####
# Requirements.
####

reqs = [
    # Our packages.
    'torch',
    'torchvision',
    'matplotlib',
    'ipython',
    'jupyter',
	'numpy',
    'pillow',
    'sklearn',
    'scipy',
]

extras = {
    'test' : [
        'pytest'
    ],
    'dev' : [
    ],
}


####
# Packages and scripts.
####

packages = find_packages(where = src_subdir)

package_data = {
    package_name: [],
}

entry_points = {
    'console_scripts': [
        'run = mnist_cnn:main',
    ],
}


####
# Import __version__.
####

project_dir = dirname(realpath(__file__))
version_file = join(project_dir, src_subdir, 'version.py')
exec(open(version_file).read())


####
# Install.
####

setup(
    name             = project_name,
    version          = __version__,
    author           = author,
    author_email     = author_email,
    url              = url,
    description      = description,
    zip_safe         = False,
    packages         = packages,
    package_dir      = {'': src_subdir},
    package_data     = package_data,
    install_requires = reqs,
    tests_require    = extras['test'],
    extras_require   = extras,
    entry_points     = entry_points,
)

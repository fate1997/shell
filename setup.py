#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="",
    description="Shell-based SBDD.",
    name='shell',
    packages=find_packages(include=['shell', 'shell.*', 'shell.*.*']),
    package_data={'': ['*.yml']},
    include_package_data=True,
    version='0.0.1',
)
"""Defines setuptools metadata."""

import setuptools
from setuptools.command.install import install

import subprocess
import os


install_requires = [line.strip() for line in open("requirements.txt").readlines()]

setuptools.setup(
    name="pixnar",
    version="0.0.1",
    author="Ads NLG",
    author_email="nlg@microsoft.com",
    description="PIXNAR NLG Repo",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    dependency_links=[
        'https://download.pytorch.org/whl/cu113'
    ],
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.0,<3.12",
)

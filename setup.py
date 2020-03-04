from setuptools import setup

setup(
    name="sarsim",
    version="0.2",
    description="Module for generating synthetic SAR/InSAR test data",
    author="Gerald Baier",
    author_email="gerald.baier@riken.jp",
    packages=["sarsim"],
    install_requires=['numpy', 'scipy']
)

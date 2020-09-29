"""AI Platform package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['h5py==2.9.0']

setup(name='attention_models',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description="""Automated prediction model for crypto-currency return based on sentiment analysis running on """
                  """Google AI Platform"""
)

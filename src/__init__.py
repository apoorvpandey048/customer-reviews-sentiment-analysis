"""
Amazon Reviews Sentiment Analysis - Multi-Task Learning
CSE3712 Big Data Analytics End-Semester Project

This package contains modules for analyzing Amazon product reviews using
multi-task learning approaches.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__course__ = 'CSE3712 Big Data Analytics'

from . import config
from . import data_loader
from . import preprocessing
from . import utils

__all__ = [
    'config',
    'data_loader',
    'preprocessing',
    'utils'
]

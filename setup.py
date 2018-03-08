"""
    Setup script for signifl.
    Copyright (C) 2018 Erik Quaeghebeur. All rights reserved.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from distutils.core import setup


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"
    ]

setup(
    name = 'signifl',
    packages = ['signifl'],
    description = 'A Python package for working with IEEE-753 binary floating point numbers that carry significance information by using a specific convention',
    url = 'https://github.com/equaeghe/signifl',
    author = 'Erik Quaeghebeur',
    author_email = 'E.R.G.Quaeghebeur@tudelft.nl',
    license = 'GPL',
    classifiers = classifiers,
    platforms = "any",
    requires= ['numpy']
)

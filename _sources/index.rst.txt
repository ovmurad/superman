Introduction
============

Welcome to **Superman**!  
This library helps you use Manifold Learning algorithms in a functional, numpy-style manner.  
It is designed to be easy to use, well-documented, and fast.


Features
--------

- Clean, Pythonic API
- Functional style
- Abstracted dense/sparse operations

Installation
------------

You can install the latest release from PyPI:

.. code-block:: bash

   pip install superman

Or install from source:

.. code-block:: bash

   git clone https://github.com/ovmurad/superman
   cd superman
   pip install -e.


Here’s a minimal example to get you started:

.. code-block:: python

    from src.geometry import Points
    from src.data import load_swiss_roll

    #returns tuple of points and intrinsic parameter
    points_coords = load_swiss_roll(1000)
    points = points_coords[0]
    coords = points_coords[1]

    #get the pairwise distance matrix with cityblock metric
    dist = points.pairwise_distance(dist_type="cityblock")
    print(dist)

Next Steps
----------

- :doc:`usage` — learn how to use MyProject in more detail  
- :doc:`modules` — explore the full API reference

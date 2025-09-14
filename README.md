# superman

## Installation

1. Install poetry. Instructions [here](https://python-poetry.org/docs/#installation).
2. Clone this repo with your preferred method.
3. Using the command line, navigate to the repo and run `poetry install`.

## Docs

Docs are located at https://ovmurad.github.io/superman/.

## Overall

## Project Management

## Structure

### data

### notebooks

### src

### tests

## Main Package Components

### array

This submodule will implement a common interface for dense numpy array and sparse scipy.sparse.coo_array.
Thus it will, generally, implement the same operations(as needed by upstream computations) for both dense arrays and
sparse arrays.
The common interface will be implemented in the Array Class, while the dense and sparse implementations will be
implemented by the DenseArray and SparseArray classes respecively.

#### BaseArray

Abstract class that contains functionality that all children must implement. Things like `__add__` and `__iadd__` are defined

#### DenseArray

Class that performs operations on a `Storage` field. Copies the storage if operation is out-of-place, 
otherwise modifies the storage directly. Currently uses numpy operations.

#### Storage

Class that uses some `BACKEND` to execute Dense operations.

#### Backend

Implements the Dense operations. Currently uses numpy operations.

#### SparseArray

Class that uses scipy's `coo_array` to exectue Sparse operations.

### object

This submodule contains mixin classes for the generic `Object` and all non-implementation mixin classes.
Mixins are all abstract and are only used as mixins.

#### Object

This submodule contains mixin classes for the generic `Object` and all non-implementation mixin classes.
`Objects` inherit a metadata field and some `fixed_ndim` and `fixed_dtype` that will be checked against 
when an instance of that `Object` is created. If the constructor receives array-like data that has ndim 
not equal to `fixed_ndim` or dtype not equal to `fixed_dtype`, it will error.

Generally, all child `Object` implementations will inherit both a mixin class and an Array class.
Thus, all implementations are the same abstract mixin class and have functionality from the Array class.
You can expect the functionality defined in the mixin class without worrying about the specific implementation.

All child Objects must inherit from the mixin FIRST and then its array. This is very important because 
the flow of constructor data follows the MRO and the data must terminate in the BaseArray class.

##### Common interface

A very rough list of Objects with their dimensions, type of Array Object supported, and description.

#### Function

- **Variable Name**: `f`.
- **Array Type**: Dense.
- **Dimensions**: `npts` for univariate functions/coords; (`npts`, `k`) for multivariate functions.
- **Description**: Array containing the values of a function at the set of points we're working with.
  The array can have only one dimension(a function to $\mathbb{R}$) or multiple dimensions(a function to
  $\mathbb{R}^{nparams}$).
  This class is going to be essential for visualization and will generally represent the colors by which we color the
  points.

### geometry

This submodule contains all geometry-related objects. Matrix objects and `Points` are included.

#### Points/Embedded Points/Point Coordinates

- **Variable Name**: for points in ambient space - `xamb`, `yamb`, `amb`; for embedded points - `xemb`, `yemb`, `emb`;
  for point coordinates - `xcoord`, `ycoord`, `coord`;
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`) for points in ambient space; (`npts`, `p`) for embedded points; (`npts`, `d`) for point
  coordinates.
- **Description**: Array containing a set of points sampled from a manifold, subsequently embedded by some embedding
  algorithm or function, or the coordinates of the points in some chart.

#### Distance Matrix

- **Variable Name**: `dist`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).
- **Description**: Matrix object that represents pairwise distances. All pairwise distances are calculated, resulting in a fully dense matrix.
  Thresholding is highly recommended.

#### Affinity Matrix

- **Variable Name**: `aff`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).
- **Description**: Matrix object that represents similarities. Reweights a distance matrix's edges with some function like a gaussian.

#### Laplacian Matrix

- **Variable Name**: `lap`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).
- **Description**: Matrix object that represents a graph laplacian. Computes graph laplacian variations on an affinity matrix.

#### Laplacian Spectrum

- **Varaible Name**: `lap_evals`.
- **Array Type**: Dense.
- **Dimensions**: `p`.
- **Description**: TODO.

#### Tangent Bundle

- **Variable Name**: `tan_bun`.
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`, `d`) in ambient space; (`npts`, `p`, `d`) in embedding space.
- **Description**: TODO.

#### Local Covariance Matrices

- **Variable Name**: `cov_mat`.
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`, `D`) in ambient space; (`npts`, `p`, `p`) in embedding space.
- **Description**: TODO.

#### Local Covariance Spectra

- **Variable Name**: `cov_evals`.
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`) in ambient space; (`npts`, `p`) in embedding space.
- **Description**: TODO.

#### Metric/Co-Metric

- **Variable Name**: `met`, `cmet`.
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`, `D`) in ambient space and coordinates; (`npts`, `p`, `p`) in embedding space and
  coordinates; (`npts`, `d`, `d`) in intrinsic coordinates.
- **Description**: TODO.

#### Metric/Co-Metric Spectra

- **Variable Name**: `met_evals`, `met_evecs`.
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`, `D`) in ambient space and coordinates; (`npts`, `p`, `p`) in embedding space and
  coordinates; (`npts`, `d`, `d`) in intrinsic coordinates.
- **Description**: TODO.

### linalg

## Secondary Package Components

### utils

### io

### data

This submodule contains utilities for generating and loading datasets into Superman objects.

### sampling


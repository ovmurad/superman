# superman

## Installation

1. Install poetry. Instructions [here](https://python-poetry.org/docs/#installation).
2. Clone this repo with your preferred method.
3. Using the command line, navigate to the repo and run `poetry install`

Note: docs are located at https://ovmurad.github.io/superman/

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

#### Array

##### Common Interface

- Common Array properties: shape, num, nnz, etc.

#### DenseArray

##### Common Interface Specifics

##### Specific Interface

#### SparseArray

##### Interface Specifics

##### Specific Interface

### object

This submodule will implement the interface with the Array Class depending on the type of 'geometric' object wrapping
the array.
Abstractly, each Object subclass will implement the 'semantic meaning' of the array it manipulates.
Some examples include: **Points**(these will be points in either the ambient or embedding space, always Dense with size
npts x nfeats), **Laplacian**(handles laplacian matrices, can be either Dense or Sparse of size npts x npts), and many
others.
Some of these objects will support only DenseArrays or SparseArrays, but some will support both, making the common
interface in Array necessary.
The Object class should implement some common functions to all subclasses. Wrappers for the underlying Array object and
a `visualize` function would be a good place to start, but others might be useful.

#### Object

##### Common interface

A very rough list of Objects with their dimensions, type of Array Object supported, and description.

#### Points/Embedded Points/Point Coordinates

- **Variable Name**: for points in ambient space - `xamb`, `yamb`, `amb`; for embedded points - `xemb`, `yemb`, `emb`;
  for point coordinates - `xcoord`, `ycoord`, `coord`;
- **Array Type**: Dense.
- **Dimensions**: (`npts`, `D`) for points in ambient space; (`npts`, `p`) for embedded points; (`npts`, `d`) for point
  coordinates.
- **Description**: Array containing a set of points sampled from a manifold, subsequently embedded by some embedding
  algorithm or function, or the coordinates of the points in some chart.

#### Function

- **Variable Name**: `f`.
- **Array Type**: Dense.
- **Dimensions**: `npts` for univariate functions/coords; (`npts`, `k`) for multivariate functions.
- **Description**: Array containing the values of a function at the set of points we're working with.
  The array can have only one dimension(a function to $\mathbb{R}$) or multiple dimensions(a function to
  $\mathbb{R}^{nparams}$).
  This class is going to be essential for visualization and will generally represent the colors by which we color the
  points.

#### Distance Matrix

- **Variable Name**: `dist_mat`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).

#### Affinity Matrix

- **Variable Name**: `aff_mat`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).
- **Description**: TODO.

#### Laplacian Matrix

- **Variable Name**: `lap_mat`.
- **Array Type**: Dense or Sparse.
- **Dimensions**: (`npts`, `npts`).
- **Description**: TODO.

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

### geometry

## Secondary Package Components

### utils

### io

### data

### sampling


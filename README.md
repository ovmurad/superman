# superman

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

#### Points

- **Array Type**: Dense.
- **Dimensions**: npts x nfeats.
- **Description**: Array containing a set of points sampled from a manifold or subsequently embedded by some embedding
  algorithm or function.

#### Params

- **Array Type**: Dense.
- **Dimensions**: npts or npts x nparams.
- **Description**: Array containing the values of a function or of a parameter(like a coordinate or generative
  parameter) at the set of points in question.
  The array can have only one dimension(a function to $\mathbb{R}$) or multiple dimensions(a function to
  $\mathbb{R}^{nparams}$).
  This class is going to be essential for visualization and will generally represent the colors by which we color the
  points.

#### Distance Matrix

- **Array Type**: Dense or Sparse.
- **Dimensions**: npts x npts.

#### Affinity Matrix

- **Array Type**: Dense or Sparse.
- **Dimensions**: npts x npts.

#### Laplacian Matrix

- **Array Type**: Dense or Sparse.
- **Dimensions**: npts x npts.

#### Tangent Bundle

- **Array Type**: Dense.
- **Dimensions**: npts x nfeats x d.

#### Metric

- **Array Type**: Dense or Sparse.
- **Dimensions**: npts x nfeats x nfeats or npts x d x d.

### linalg

### Covariance Matrix

- **Array Type**: Dense.
- **Dimensions**: nfeats or d.

### Spectral Embedding

- **Array Type**: Dense or Sparse.
- **Dimensions**: 2 x npts
- **Description**: Tuple containing eigenvalues and eigenvectors

### geometry

## Secondary Package Components

### utils

### io

### data

### sampling


# Pyglrm

Pyglrm is a python package for modeling and fitting generalized low rank models (GLRMs), based on the Julia package LowRankModels.jl. 

GLRMs model a data array by a low rank matrix, and include many well known models in data analysis, such as principal components analysis (PCA), matrix completion, robust PCA, nonnegative matrix factorization, k-means, and many more.

For more information on GLRMs, see [our paper][glrmpaper].

## Requirements
- OS: Any linux or MacOS
- Python (2.7 or 3.x)
- Working installation of git

This package has been tested on Ubuntu linux and may not work on non-linux OSes.

This package also relies on Julia.  On linux systems this package can install the most recent version of Julia automatically.  The version of Julia distributed by most linux distributions is may not be recent enough to run this package.  We recommend you use the official binary from [the Julia website](https://julialang.org/downloads/).

**Note** If you use the version of Julia installed by this package, you may need to run

```
export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
```

in order to access Julia.


## Installation

### Windows

Windows based installations are not supported yet.

### MacOS Installation

3.  Install the most recent version of Julia (0.6) by following downloading the appropriate installer from [the Julia website](https://julialang.org/downloads/) and following the direction for your operating system on the [instructions page](https://julialang.org/downloads/platform.html).
3.  Check that Julia runs on the command line by running the command ```julia``` on the command line.
3.  Using your choice of ```pip```, ```pip2```, or ```pip3``` depending on the version of Python you intend on using, run the command
    ```
    pip install git+https://gitlab.datadrivendiscovery.org/Cornell/pyglrm
    ```
    
    The installation will get the package via git - you may need to enter you password for gitlab.

### Linux

3.  Note that the default distribution of Julia included in most package managers is not sufficiently up to date to run this package.  We instead using the version of Julia from the Julia website.  The installer for this package can install Julia for you.
3.  Using your choice of ```pip```, ```pip2```, or ```pip3``` depending on the version of Python you intend on using, run the command
    ```
    pip install git+https://gitlab.datadrivendiscovery.org/Cornell/pyglrm
    ```
    
    The installation will get the package via git - you may need to enter you password for gitlab.
3.  If you let pip install Julia, you may need to run the command
    ```
    export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
    ```

## Common Troubleshooting

3.  Segmentation faults

    The underlying software that runs the package compiles itself for one version of Python at a time.  For example, if you install the package using Python 2.7 and then use Python 3.6 you will get a segmentation fault.
    
    If switching between versions of Python is your problem, there is a simple solution.  Each time you switch version of Python first run
    ```
    whereis python
    whereis python3
    ```
    or
    ```
    which python
    which python3
    ```
    to find the absolute path to the version of Python you plan to use.  Then run the following commands in Julia
    
    ```
    ENV["PYTHON"] = "/path/to/python/binary"
    Pkg.build("PyCall")
    exit()
    ```
    
    This should resolve the issue.

3.  On linux, after installation "Julia" cannot be found.

    You likely need to run the command
    ```
    export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
    ```
    

## Generalized Low Rank Models

GLRMs form a low rank model for tabular data `A` with `m` rows and `n` columns,
which can be input as an array or any array-like object (for example, a data frame).
The desired model is specified by choosing a rank `k` for the model,
an array of loss functions `losses`, and two regularizers, `rx` and `ry`.
The data is modeled as `X'*Y`, where `X` is a `k`x`m` matrix and `Y` is a `k`x`n` matrix.
`X` and `Y` are found by solving the optimization problem
<!--``\mbox{minimize} \quad \sum_{(i,j) \in \Omega} L_{ij}(x_i y_j, A_{ij}) + \sum_{i=1}^m r_i(x_i) + \sum_{j=1}^n \tilde r_j(y_j)``-->

    minimize sum_{(i,j) in obs} losses[j]((X'*Y)[i,j], A[i,j]) + sum_i rx(X[:,i]) + sum_j ry(Y[:,j])

The basic type used by LowRankModels.jl is the GLRM. To form a GLRM,
the user specifies

* the data `A` 
* the array of loss functions `losses`
* the regularizers `rx` and `ry`
* the rank `k`

Losses and regularizers must be of type `Loss` and `Regularizer`, respectively,
and may be chosen from a list of supported losses and regularizers, which include

Losses:

* quadratic loss `QuadLoss`
* hinge loss `HingeLoss`
* logistic loss `LogisticLoss`
* poisson loss `PoissonLoss`
* weighted hinge loss `WeightedHingeLoss`
* l1 loss `L1Loss`
* ordinal hinge loss `OrdinalHingeLoss`
* periodic loss `PeriodicLoss`
* multinomial categorical loss `MultinomialLoss`
* multinomial ordinal (aka ordered logit) loss `OrderedMultinomialLoss`

Regularizers:

* quadratic regularization `QuadReg`
* constrained squared euclidean norm `QuadConstraint`
* l1 regularization `OneReg`
* no regularization `ZeroReg`
* nonnegative constraint `NonNegConstraint` (eg, for nonnegative matrix factorization)
* 1-sparse constraint `OneSparseConstraint` (eg, for orthogonal NNMF)
* unit 1-sparse constraint `UnitOneSparseConstraint` (eg, for k-means)
* simplex constraint `SimplexConstraint`
* l1 regularization, combined with nonnegative constraint `NonNegOneReg`
* fix features at values `y0` `FixedLatentFeaturesConstraint(y0)`

Each of these losses and regularizers can be scaled
(for example, to increase the importance of the loss relative to the regularizer)
by having arguments like `QuadLoss(scale=1.0)` in class initializations.


## Example

For example, the following code performs PCA with `k=2` on the `3`x`4` matrix `A`:

    import numpy as np
    from pyglrm import *
    A = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 5, 6, 7]])
    k = 2 #number of target dimensions
    losses = QuadLoss()
    rx = ZeroReg()
    ry = ZeroReg()
    g = glrm(losses, rx, ry) #create a class for GLRM (Here it does PCA) 
    X, Y, ch = g.fit_transform(A, k) 
    A_pca = np.dot(np.transpose(X), Y) #result for PCA
    a_new = np.array([6, 7, 8, 9]) #A new line to be tested
    x = g.predict(a_new) #the latent representation of a_new
    

which runs an alternating directions proximal gradient method on `g` to find the
`X` and `Y` minimizing the objective function.
(`ch` gives the convergence history; see
[Technical details](https://github.com/madeleineudell/LowRankModels.jl#technical-details)
below for more information.)

The `losses` argument can also be an array of loss functions,
with one for each column (in order). For example,
for a data set with 3 columns, you could use

    losses = [QuadLoss(), LogisticLoss(), HingeLoss()]

Similiarly, the `ry` argument can be an array of regularizers,
with one for each column (in order). For example,
for a data set with 3 columns, you could use

    ry = [QuadReg(1), QuadReg(10), FixedLatentFeaturesConstraint([1.,2.,3.])]

This regularizes the first to columns of `Y` with `||Y[:,1]||^2 + 10||Y[:,2]||^2`
and constrains the third (and last) column of `Y` to be equal to `[1,2,3]`.

[More examples here.](https://gitlab.datadrivendiscovery.org/Cornell/pyglrm/tree/master/examples)


## Standard low rank models

Low rank models can easily be used to fit standard models such as PCA, k-means, and nonnegative matrix factorization.
The following functions are available:

* `pca`: principal components analysis
* `qpca`: quadratically regularized principal components analysis
* `rpca`: robust principal components analysis
* `nnmf`: nonnegative matrix factorization

To create a class for one of these standard models, replace `glrm` in the above example with the model name above. Any keyword argument valid for a `GLRM` object,
such as an initial value for `X` or `Y`
or a list of observations,
can also be used with these standard low rank models.


## Citing this package

If you use LowRankModels for published work,
we encourage you to cite the software.

Use the following BibTeX citation:

    @article{glrm,
      title = {Generalized Low Rank Models},
      author ={Madeleine Udell and Horn, Corinne and Zadeh, Reza and Boyd, Stephen},
      doi = {10.1561/2200000055},
      year = {2016},
      archivePrefix = "arXiv",
      eprint = {1410.0342},
      primaryClass = "stat-ml",
      journal = {Foundations and Trends in Machine Learning},
      number = {1},
      volume = {9},
      issn = {1935-8237},
      url = {http://dx.doi.org/10.1561/2200000055},
    }

[glrmpaper]: https://people.orie.cornell.edu/mru8/doc/udell16_glrm.pdf

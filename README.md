This repository implements an example of the Dimensionality Reduction API
developed for DARPA's D3M program.

Documentation available [here](https://docs.google.com/document/d/1kc3uyOzx7S4aoMy0KN-QE__8Mxazr5tPObV4Sr1rH2g/edit#heading=h.qaq7osingv87).

Requirements:
- git must be installed
- Python 2.7
- OS: Any linux

This package has been tested on Ubuntu linux and may not work on non-linux OSes.

This package also relies on Julia.  On linux systmes the most recent version of this can be automatically installed.  The version of Julia distributed by most linux distributions is not sufficient to run this package, rather the official binary from [the Julia website](https://julialang.org) is recommended.

**Note** If you use the version of Julia installed by this package, you may need to run

```
export PATH=$PATH:$HOME/.julia/julia-903644385b/bin
```
in order to access Julia.

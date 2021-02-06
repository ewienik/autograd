# autograd

Sample framework for automated differentiation algorithm

# Artifacts

- `ag_test`: unit test of frameworks
- `mlc`: sample MNIST linear classifier

# Compilation tests

These steps are tested in Ubuntu 20.04 2021-02-06:

```
$ unzip autograd
$ mkdir release
$ cd release
$ CC=gcc-10 CXX=g++-10 cmake ../autograd -GNinja -DCMAKE_BUILD_TYPE=Release
$ ninja
$ ./ag_test # run unit tests
$ ./mlc ../autograd/ext/mnist # run sample MNIST linear classifier
```


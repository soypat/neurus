# neurus
A repository to show the most basic Neural Network implementation possible with
different functionality/optimizations.

| Complexity | Features      |
|------------|---------------|
|   Level 0  | ~100 line basic building blocks of a neural network demonstration in [`level0.go`](level0.go). Training the neural network is possible with logic in [`level0training.go`](level0training.go). One does not need to train a neural network to use it, which is why these files are split for the most basic level. Based on the first part of [Sebastian Lague's video](https://www.youtube.com/watch?v=hfMk-kjRv4c). |
|  Level 1  | *Planned...* |
| Optimized | An advanced implementation of a NN with backpropagated gradient descent using a velocity-momentum model. Runs much faster than Level 0. Based on Sebastian Lague's [final neural network implementation](https://github.com/SebLague/Neural-Network-Experiments) from the final section of his [video](https://www.youtube.com/watch?v=hfMk-kjRv4c). Still contains bugs.  |



## Mnist digit image/label database
Mnist database package available for import under [`mnist`](mnist).
![mnist](mnist/3.png).

```shell
go get github.com/soypat/neurus/mnist@latest
```
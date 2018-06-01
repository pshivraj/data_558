## l2-regularized logistic regression with both gradient descent and fast gradient descent Algorithm.

The repository implement l2-regularized logistic regression with both gradient descent and fast gradient descent Algorithm with backtracking rule.

## Project structure
Model folder contains `fastalgo.py` which impliments both gradient and fast gradient_descent.

`fastalgo.py` 
* contains function '`plot_grad_vs_fast_obj`' to compare two gradient descent methods with plots.
* contains function '`sklearn_compare_fast`' to compare results against sklearn implimentation with plots.

`simulated.py` is a demo file, launch the method on a simple simulated dataset.

`real_world.py` is a demo file, launch the method on a real world  dataset.

Required packages

Requisite packages can be installed based on requirements.txt

The code has been tested to run on python3

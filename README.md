# AM205 Final Project - Applications of Numerical Methods to LUNA

#### Contributors: Victor Avram, Michael Butler, M. Elaine Cunha, Jack Scudder

The original LUNA paper can be found [here](https://arxiv.org/abs/2006.11695).

The final report `AM205_Project_Luna.pdf` can be found under the directory **final_report**.

### Neural Network and LUNA Implementations

---

* `feed_forward.py` - Contains the base class for a neural network
* `nlm.py` - Contains the base class for the NLM model which defines a neural linear model
* `luna.py` - Contains the base class for the LUNA model; inherits from `nlm.py`
* `bayes_helpers.py` - Helper functions for Bayesian analysis within the LUNA model; contains functions for sampling from the prior or posterior, calculating the prior/posterior predictive, plotting predictive intervals, calculating
* `config.py` - Contains standardized configuration parameters for NLM and LUNA models

### NLM and LUNA Demos

---

* `LUNADemo.ipynb` - Python notebook with demonstration of LUNA model on a toy dataset
* `NLMDemo.ipynb` - Python notebook with demonstration of LUNA model on a toy dataset
* `PriorPredictives_Demo.ipynb` - Python notebook with demonstration of how regularization affects the prior and posterior predictive of an NLM

### Alternatives for Optimizers

---

* `optimizers.py` - Implementations for 4 optimization methods as listed below.
  * Steepest Descent
  * Newton's Method
  * BFGS
  * Conjugate Gradient
* `optimizer_tests.py` - Basic tests for each alternative optimizer. 3 functions are tested and are given below
  * $x^3 + 4e^y + 10z^4$
  * $x^2 + 3x + 1$
  * $x^2 y^2$
* `LUNA_optimizer_metrics.py` - Produces LUNA training peformance metrics for each optimizer
* `NLM_Metrics_Analysis.ipynb` - Produces NLM training performance metrics for each optimizer

### Finite Difference Modifications
---

#### Due to changes in the content of `luna.py`, the Finite Difference Modifications were kept on `jack_finite_diff` branch.  There you will find:

* `Scalar_Delta_Demo_3500_iter.ipynb` - Encompasses the simulation for various step sizes of 0.1, 0.001, 0.0001 at 3500 training iterations of LUNA.  Captures runtimes, and generates plots for each model.
* `Random_Indices_3500_iter.ipynb` - Encompasses the simulation for sampling percentages of observation indices at rates of 1%, 10%, 20%, 40%, 60%, and 80%.  Captures runtimes, and generates plots for each model.  Also plots log-likelihood vs. percentage of sampled indices.
* `Scalar_Delta_Demo_3500_iter.pdf` and `Random_Indices_3500_iter.pdf` - 2 pdfs that show the results of the 2 experimental notebooks
* `Scalar_Delta_Demo_v2.ipynb` and `Random_Indices_v2.ipynb` - 2 notebooks that follow the same experimental setup, but compile significantly faster due to fewer iterations.
* `luna.py` - Slightly modified from `luna.py` on `main` branch to allow for finite difference experiment inputs
* Directory **figs** - Contains plots generated by the notebooks above


### Informative Plots and Data Files
---

* Directory **figs** - Contains plots of true data, predictions, and predictive uncertainty for NLM and LUNA training examples.
* Directory **NLM_performance_data** - Contains plots of true data, predictions, and predictive uncertainty as well as descriptive performance metrics for NLM training across alternative optimization methods.
* Directory **LUNA_performance_data** - Contains plots of true data, predictions, and predictive uncertainty as well as descriptive performance metrics for LUNA training across alternative optimization methods.
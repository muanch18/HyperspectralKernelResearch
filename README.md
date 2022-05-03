## Hyperspectral Kernel Research
Working under a professor at the Courant Institute of Mathematics, I researched methods on how to speed up kernels methods for large scale computation.

### Hyperspectral Data
To simulate large scale computation, I used hyperspectral data for the purpose of data with a large number of dimensions and hundreds of datapoints

### Kernels
Naive, linear classifiers often misclassify points in a nonlinear dataset. Kernels are able to implicility and efficiently lift data to a new, scalable dimension, so that a new 
classification model can be used that will more accurately classify the points. 

### Script
After researching feature engineering and common syntax in Tensorflow 2.0, I formulated a Python Script to initialize a Kernel Support Vector Machine that can classify data points in the
hyperspectral dataset. 

### Research
Using the Python Script, I answered 4 different questions. 
1. How does the Training Test Size affect the Training Time?
   - Hypothesis: Square function correlation
   - Result: Square function correlation; An increase in training test proportion size exponentially increased the training time. 
3. How does the Inaccuracy depend on the Proportion of Training Test Data?
   - Hypothesis: Inverse Square function correlation
   - Result: Natural log function correlation
5. How does the Training Test Size affect the Prediction Time?
   - Hypothesis: Polynomial function correlation
   - Result: Negative Coefficient, Polynomial function of Order 2 correlation; the max prediction time took place at 60-70% training data. 
7. How does the Kernel Type affect the Inaccuracy Percentage?
   - Tested the Gaussian, Laplace, Polynomial, and Linear Kernel function against each other. The Polynomial Kernel had the lowest Inaccuracy percentage. 


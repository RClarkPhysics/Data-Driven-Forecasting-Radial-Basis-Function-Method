# Data-Driven-Forecasting-Radial-Basis-Function-Method
The Radial basis functions can take many forms. So far, I have favored using the Gaussian form the of the Radial Basis Function (RBF) and the Multi Quadratic form for DDF. Both folders include code and example results. The Gaussian folder has two python scripts; their only difference is how R is chosen as a paramter in the gaussian. The base form of the RBF sets the same R for all RBF in the expansion, whereas RBF_NN uses the nearest neighbors to estimate how how large R should be as well as a normalization procedure to keep the size of all the dimensions equal to magnitude 1; the intuition for this is that centers densely packed in, should have a smaller range of influence, and centers that are more sparesely seperated will need a larger range of influence, so they should have different R values based on their nearest neighbors.
I also include a jupyter notebook used for generatingn the data sets I train and predict with.


DDF Basics: Here is a quick refresher on what DDF is doing and what the code is trying to accomplish. Starting with the dynamical equations, we have some system that has the following differential equations

dx(t)/dt = F(x(t))

We want to model the behavior of the observed variable x(t), but F(x(t)) is unkown to us. We can approximate the problem with the Euler Formula

x(n+1) = x(n) + dt*F(x(t))

Now we want a Function Representation for F(x(t)). Inspired by our knowledge of NaKL, we choose a representation the form:

f(x(t)) = sum(RBF(V(n),c_q))

We use these two equations above to write down a cost function to fit our coefficients in the RBF's:

Minimize sum_length [(x(n+1)-x(n)) - sum(RBF(V(n),c_q))]^2

Because the function representation is linear in the coefficients, we can rewrite the formula in terms of W*X where W are the weights, and X is the value of either the RBF. This minimization problem can be solved with Ridge Regression:

W = YX^T(XX^T)^-1

[Y] = 1 x Time

[X] = Parameter Length x Time

with the minizimation done, f(x(t)) can now be used to forecast forward in time.

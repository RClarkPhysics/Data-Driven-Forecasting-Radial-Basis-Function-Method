# Data-Driven-Forecasting-Radial-Basis-Function-Method
The Radial basis functions can take many forms. So far, I have favored using the Gaussian form the of the Radial Basis Function (RBF) and the Multi Quadratic form for DDF. Both folders include code and example results. The Gaussian folder has two python scripts; their only difference is how R is chosen as a paramter in the gaussian. The base form of the RBF sets the same R for all RBF in the expansion, whereas RBF_NN uses the nearest neighbors to estimate how how large R should be; the intuition for this is that centers densely packed in, should have a smaller range of influence, and centers that are more sparesely seperated will need a larger range of influence, so they should have different R values based on their nearest neighbors.
I also include a jupyter notebook used for generatingn the data sets I train and predict with.

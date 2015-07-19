# HighOrderRegression
Linear regression with high Order Polynomial
This class template is used to smooth a numerical function.
You need MKL/LAPACK to compile it.
The test directory contains a function.txt which is the numerical function with some cusps.
And smoothfunciton.txt which is obtained via 10th order linear regression. 
Since at 10th order you have a good variance which doens't change much if you go to a higher order.

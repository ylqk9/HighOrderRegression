// build.win.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "regression.h"


int main()
{
	grid<double> box(200, 0.4);
	Regression<double> myregression(box);
	myregression.LoadFunction("function.txt");
	//generate variance vs order, find best order
	myregression.Variance();
	//use the best order for the regression, 10 in the test case
	myregression.Solve(10);
	//run prediction
	myregression.Predict();
	ofstream outfile("smoothfunction.txt");
	for(int i = 0; i < 200; ++i) outfile << myregression.y_reg[i] << endl;
	return 0;
}


#include "Regression.h"
#include <iostream>

using namespace std;



Regression::Regression(const int &it, const int &s, const int &nf,
	const double &a, const double &atol, double *y, double **x): 
	iter_max(it), samples(s), numFeatures(nf), alpha(a), 
	abs_tol(atol), yi(y), xi(x)
{
}


Regression::~Regression()
{
	for (int i = 0; i < samples; i++) delete[] xi[i];
	delete[] xi, yi;
}


Regression::Regression(const Regression & r): weights(new double[r.numFeatures]),
												iter_max(r.iter_max),
												samples(r.samples),
												numFeatures(r.numFeatures),
												alpha(r.alpha),
												abs_tol(r.abs_tol),
												xi(new double*[r.samples]),
												yi(new double[r.samples])
{
	for (int i = 0; i < samples; i++)
	{
		yi[i] = r.yi[i];
		xi[i] = new double[numFeatures];
		for (int j = 0; j < numFeatures; j++) xi[i][j] = r.xi[i][j];
	}

	for (int i = 0; i < numFeatures; i++) weights[i] = r.weights[i];
}


void Regression::gradientDescent()
{
	int iter = 0;
	bool not_converged = false;
	double old_cost = 1, new_cost = 0, sum2 = 0;
	double k = alpha / samples;

	// Initialise weights
	weights = new double[numFeatures];
	for (int i = 0; i < numFeatures; i++) weights[i] = 0.0;

	while (abs(new_cost - old_cost) > abs_tol)
	{
		if (iter >= iter_max)
		{
			cerr << "Could not reach convergence!";
			not_converged = true;
			break;
		}

		for (int j = 0; j < numFeatures; j++)
		{
			sum2 = 0;
			for (int i = 0; i < samples; i++)
			{
				sum2 += (hypothesis(xi[i]) - yi[i]) * xi[i][j];
			}
			weights[j] = weights[j] - k * sum2;
		}

		old_cost = new_cost;
		new_cost = cost_function();

		if (0 == iter % 100)
		{
			cout << "Descending...		iter: " << iter << "   \t";
			cout << "Error: " << new_cost << endl;
		}

		iter++;
	}
	if (not_converged == false) cout << "Convergence achieved" << endl;

}


double Regression::hypothesis(double *x)
{
	double hyp = 0;

	for (int i = 0; i < numFeatures; i++)
	{
		hyp += x[i] * weights[i];
	}
	return hyp;
}


double * Regression::Get_weights()
{
	double * w = new double[numFeatures];
	for (int i = 0; i < numFeatures; i++)
	{
		w[i] = weights[i];
	}
	return w;
}


double Regression::cost_function()
{
	double cost = 0.0;

	for (int i = 0; i < samples; i++)
	{
		cost += pow(hypothesis(xi[i]) - yi[i], 2.0);
	}

	cost = cost * (1.0 / (2.0 * samples));

	return cost;
}

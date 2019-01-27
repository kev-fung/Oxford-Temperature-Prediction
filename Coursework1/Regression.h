#pragma once
class Regression
{
public:
	Regression(const int &it, const int &s, const int &nf,
		const double &a, const double &atol, double *y, double **x);
	~Regression();
	Regression(const Regression &r);

	void gradientDescent();
	double cost_function();
	double hypothesis(double *x);

	double *Get_weights();

private:
	const int iter_max;
	const int samples;
	const int numFeatures;
	const double alpha;
	const double abs_tol;
	double *weights;
	double *yi;
	double **xi;
};


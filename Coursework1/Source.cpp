
/* ACSE 5 -
	Langrange Interpolation
*/

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <vector>

#include "Data.h"
#include "Regression.h"


using namespace std;

void printvector(vector <vector <string> > data)
{

	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].size(); j++)
			cout << data[i][j] << " ";
		cout << data[i].size() << endl;
	}
}


void delete_double(int samples, double **input_doubles)
{
	for (int i = 0; i < samples; i++) {
		delete[] input_doubles[i];
	}

	delete[] input_doubles;
}


void print_double(int num_rows, int num_columns, double **input_doubles)
{
	for (int i = 0; i < num_rows; i++)
	{
		for (int j = 0; j < num_columns; j++)
		{
			cout << input_doubles[i][j] << ", ";
		}
		cout << "	" << i << endl;
	}
}


void fileoutput(string name, const int &samples, double *y, double *x)
{
	ofstream datafile(name);
	if (datafile.is_open()) {
		for (int i = 0; i < samples; i++) {
			datafile << x[i] << "\t" << y[i] << endl;
		}
		datafile.close();
	}
	else {
		cerr << "Unable to open file";
	}

}


void plotdata(FILE *gnuplotPipe, string testfile, string tname, string xlab, string ylab, string title, int tstart, int tend, bool save)
{
	if (gnuplotPipe) {   // If found gnuplot:
		fprintf(gnuplotPipe, "set xlabel '%s'\n", xlab.c_str());
		fprintf(gnuplotPipe, "set ylabel '%s'\n", ylab.c_str());
		fprintf(gnuplotPipe, "set title '%s'\n", title.c_str());
		fprintf(gnuplotPipe, "set grid\n");
		fprintf(gnuplotPipe, "set size 1, 1\n");

		//Plotting both our input and prediction data:
		fprintf(gnuplotPipe, "plot [%d : %d] [-5 : 25] \"input.dat\" using 1:2 title 'Input' \n", tstart, tend);
		fprintf(gnuplotPipe, "replot \"%s\" using 1:2 title '%s' \n", testfile.c_str(), tname.c_str());

		// Saving plot to file:
		if (save == true) {
			fprintf(gnuplotPipe, "set terminal push\n"); // save original format setting
			fprintf(gnuplotPipe, "set terminal gif size 640, 480\n"); // set jpg size
			fprintf(gnuplotPipe, "set output \"%s.jpg\"\n", tname.c_str()); // set jpg name
			fprintf(gnuplotPipe, "replot\n");
			fprintf(gnuplotPipe, "set terminal pop\n"); // restore original format setting
		}

		fflush(gnuplotPipe); //flush pipe

		fprintf(gnuplotPipe, "\nexit \n");
		_pclose(gnuplotPipe);    //close pipe
	}
}


void plotsingledata(FILE *gnuplotPipe, string testfile, string tname, string xlab, string ylab, string title, bool save)
{
	if (gnuplotPipe) {   // If found gnuplot:
		fprintf(gnuplotPipe, "set xlabel '%s'\n", xlab.c_str());
		fprintf(gnuplotPipe, "set ylabel '%s'\n", ylab.c_str());
		fprintf(gnuplotPipe, "set title '%s'\n", title.c_str());
		fprintf(gnuplotPipe, "set grid\n");
		fprintf(gnuplotPipe, "set size 1, 1\n");

		//Plotting both our input and prediction data:
		fprintf(gnuplotPipe, "plot \"%s\" using 1:2 title '%s' \n", testfile.c_str(), tname.c_str());

		// Saving plot to file:
		if (save == true) {
			fprintf(gnuplotPipe, "set terminal push\n"); // save original format setting
			fprintf(gnuplotPipe, "set terminal gif size 640, 480\n"); // set jpg size
			fprintf(gnuplotPipe, "set output \"%s.jpg\"\n", tname.c_str()); // set jpg name
			fprintf(gnuplotPipe, "replot\n");
			fprintf(gnuplotPipe, "set terminal pop\n"); // restore original format setting
		}

		fflush(gnuplotPipe); //flush pipe

	}
}


double f_max(const int &samples, double *x)
{
	double max = -1e9;
	for (int i = 0; i < samples; i++)
	{
		// Find max:
		if (x[i] > max) max = x[i];
	}
	return max;
}


double f_min(const int &samples, double *x)
{
	double min = 1e9;
	for (int i = 0; i < samples; i++)
	{
		// Find min:
		if (x[i] < min) min = x[i];
	}
	return min;
}


double mean_norm(const int &samples, double *x)
{
	double mean = 0.0;
	double max;
	double min;

	for (int i = 0; i < samples; i++)
	{
		// Find mean - summation term:
		mean += x[i];
	}

	max = f_max(samples, x);
	min = f_min(samples, x);
	mean = mean / samples;

	// mean normalisation:
	for (int m = 0; m < samples; m++)
	{
		x[m] = (x[m] - mean) / (max - min);
	}

	return mean;
}


void norm_f(int drows, double **xi)
{
	// Selection of features to be mean-normalised:
	double *af = new double[drows];
	double *af2 = new double[drows];
	double *sh = new double[drows];
	double *shd = new double[drows];
	double *afd = new double[drows];
	double *afd2 = new double[drows];

	for (int i = 0; i < drows; i++)
	{
		af[i] = xi[i][5];
		af2[i] = xi[i][6];
		sh[i] = xi[i][7];
		shd[i] = xi[i][8];
		afd[i] = xi[i][9];
		afd2[i] = xi[i][10];
	}

	mean_norm(drows, af);
	mean_norm(drows, af2);
	mean_norm(drows, sh);
	mean_norm(drows, shd);
	mean_norm(drows, afd);
	mean_norm(drows, afd2);

	for (int i = 0; i < drows; i++)
	{
		xi[i][5] = af[i];
		xi[i][6] = af2[i];
		xi[i][7] = sh[i];
		xi[i][8] = shd[i];
		xi[i][9] = afd[i];
		xi[i][10] = afd2[i];
	}

	delete[] af, af2, sh, shd, afd, afd2;
}


int main()
{
	// ----- Data Processing ------
	string file_name = "oxforddata.txt";
	const int numFeatures = 10 + 1; // 10 features + intercept term.

	Data dat(file_name, numFeatures);
	dat.Read_data();
	dat.Convert_data();
	dat.Select_features();
	dat.Select_temp();

	// ------ Correlative Feature Analysis ------
	// Output feature data before mean-normalisation:

	// Plot data with gnuplot:
	// Depending on where gnuplot is installed, you may need to change the below address:
	FILE *gnuplotPipe = _popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot\" -persist", "w");  // Open a pipe to gnuplot

	cout << "Plotting correlative data to show analysis..." << endl;

	// Get data from our object:
	const int samples = dat.Get_nsamples();
	const int dcols = dat.Get_dcols();
	double *yi = dat.Get_y();
	double **xi = dat.Get_x();
	double **data = dat.Get_data();

	// Temp pointer arrays:
	double *af1 = new double[samples];
	double *rainfall = new double[samples];
	double *sunhours = new double[samples];
	for (int i = 0; i < samples; i++)
	{
		af1[i] = data[i][4];
		rainfall[i] = data[i][5];
		sunhours[i] = data[i][6];
	}
	fileoutput("af_vs_t.dat", samples, yi, af1);
	plotsingledata(gnuplotPipe, "af_vs_t.dat", "Airfrost days", "Time of Year (Yr)", "Airfrost (D)", "Airfrost days vs time", false);
	system("pause");

	fileoutput("rf_vs_t.dat", samples, yi, rainfall);
	plotsingledata(gnuplotPipe, "rf_vs_t.dat", "Rainfall", "Time of Year (Yr)", "Rainfall (mm)", "Rainfall vs time", false);
	system("pause");

	fileoutput("sh_vs_t.dat", samples, yi, sunhours);
	plotsingledata(gnuplotPipe, "sh_vs_t.dat", "Sunhours", "Time of Year (Yr)", "Sunhours (hr)", "Sunhours vs time", false);
	system("pause");

	delete[] af1, rainfall, sunhours;
	cout << "So we know that rainfall has no corellation at all\n but we can take advantage of using sunhours and airfrost days" << endl;
	cout << endl;
	system("pause");


	//------ Pre Processing ------
	// Mean-Normalise selected feature columns:
	dat.Norm_features();
	yi = dat.Get_y();
	xi = dat.Get_x();
	data = dat.Get_data();


	// ------ Linear Regression ------
	// Learning rate
	const double alpha = 0.1;
	// Tolerance for convergence
	const double abs_tol = 1e-3;
	// Maximum iterations allowed
	const int iter_max = 1e7;

	Regression reg(iter_max, samples, numFeatures, alpha, abs_tol, yi, xi);

	// Train function i.e. compute new weights:
	cout << "Performing gradient descent: " << endl;
	reg.gradientDescent();

	// Check our cost i.e. error in the new function:
	double cost = reg.cost_function();
	cout << "\nError of our found function: " << cost << endl;

	// Display weights:
	double *weights = reg.Get_weights();
	cout << "\nFound weights for our polnomial: " << endl;
	for (int i = 0; i < numFeatures; i++) cout << weights[i] << ", ";
	cout << endl;


	// ------ Post Processing ------
	// Assign new date array for plotting:
	double *date = new double[samples];
	for (int i = 0; i < samples; i++) date[i] = (data[i][0]) + (data[i][1] / 12.0);

	// Output original avg temperature:
	fileoutput("input.dat", samples, yi, date);

	// Feed original x data and see what the found y is for comparison:
	double *test_y = new double[samples];
	for (int i = 0; i < samples; i++) test_y[i] = reg.hypothesis(xi[i]);
	fileoutput("test.dat", samples, test_y, date);


	// Select year range for viewing:
	int startyear = 2010;
	int endyear = 2018;
	cout << "\nPlotting our prediction using sample data: " << endl;
	cout << "Select the year range to view. 1929 - 2018 (Recommend range of ten years)" << endl;
	cout << "Input start year: ";
	cin >> startyear;
	cout << "\nInput end year: ";
	cin >> endyear;
	cout << endl;

	plotdata(gnuplotPipe, "test.dat", "Prediction", "Time of Year (Yr)", "Avg Temp (Deg)", "Predicting Avg Temp with Regression", startyear, endyear, false);


	// ------ Predicting Jan 2020, Jan 2030, Jan 2050 ------
	// ------ Location: Oxford 

	cout << "Predicting temperature of years 2020, 2030, 2050: ";
	system("pause");

	// Select out the time column of all samples: 
	dat.Select_only_time();
	double **ti = dat.Get_ti();

	// Arrays used to store features for single linear regression:
	double t1[2] = { 1.0, 2020 + (1.0 / 12.0) };
	double t2[2] = { 1.0, 2030 + (1.0 / 12.0) };
	double t3[2] = { 1.0, 2050 + (1.0 / 12.0) };

	// Below array used for main prediction:
	double pred_t[3] = { 2020 + (1.0 / 12.0), 2030 + (1.0 / 12.0), 2050 + (1.0 / 12.0) };

	// Normalise the time column of samples:
	double mean = 0, max = -1e9, min = 1e9;
	double *timetemp = new double[samples];

	// Before normalising, rearrange ti array to be become a 1d array:
	for (int i = 0; i < samples; i++) timetemp[i] = ti[i][1];

	mean = mean_norm(samples, timetemp);

	// Find our max and min values in time column of samples:
	max = f_max(samples, timetemp);
	min = f_min(samples, timetemp);

	for (int i = 0; i < samples; i++) ti[i][1] = timetemp[i];
	delete[] timetemp;

	// Mean normalise our prediction times as well!
	t1[1] = (t1[1] - mean) / (max - min);
	t2[1] = (t2[1] - mean) / (max - min);
	t3[1] = (t3[1] - mean) / (max - min);


	// Use Linear Regression to find Airfrost at those dates:
	cout << "\n\nPredicting Airfrost: " << endl;
	dat.Select_only_af();
	double *af = dat.Get_af();

	Regression af_reg(iter_max, samples, 2, 0.001, abs_tol, af, ti);
	af_reg.gradientDescent();

	double *pred_af = new double[3];
	pred_af[0] = af_reg.hypothesis(t1);
	pred_af[1] = af_reg.hypothesis(t2);
	pred_af[2] = af_reg.hypothesis(t3);

	cout << endl;
	for (int i = 0; i < 3; i++) cout << "Date: " << pred_t[i] << "\tPredicted AirFrost: " << pred_af[i] << endl;


	// Use Linear Regression to find Sunhours at those dates:
	cout << "\nPredicting Sunhours: " << endl;
	dat.Select_only_sh();
	double *sh = dat.Get_sh();

	Regression sh_reg(iter_max, samples, 2, 0.001, abs_tol, sh, ti);
	sh_reg.gradientDescent();

	double *pred_sh = new double[3];
	pred_sh[0] = sh_reg.hypothesis(t1);
	pred_sh[1] = sh_reg.hypothesis(t2);
	pred_sh[2] = sh_reg.hypothesis(t3);

	cout << endl;
	for (int i = 0; i < 3; i++) cout << "Date: " << pred_t[i] << "\tPredicted Sunhours: " << pred_sh[i] << endl;


	// Set up an array to contain predictive samples:
	double **pred_x = new double*[3];
	for (int i = 0; i < 3; i++) pred_x[i] = new double[11];

	// Modelling our features:
	for (int i = 0; i < 3; i++)
	{
		pred_x[i][0] = 1.0;								// Intercept term
		pred_x[i][5] = pred_af[i];						// Airfrost days
		pred_x[i][6] = pow(pred_af[i], 2.0);			// Airfrost days ^2
		pred_x[i][7] = pred_sh[i];						// Sun hours
		pred_x[i][8] = pred_t[i] * pred_sh[i];			// Interaction - Time x Sunhours
		pred_x[i][9] = pred_t[i] * pred_af[i];			// Interaction - Time x Airfrost 
		pred_x[i][10] = pred_t[i] * pred_x[i][6];			// Interaction - Time x Airfrost^2

		// Continuous date/time modelled as sine wave:
		pred_x[i][1] = sin(2 * M_PI * pred_t[i]);		// First Harmonic
		pred_x[i][2] = cos(2 * M_PI * pred_t[i]);
		pred_x[i][3] = sin(4 * M_PI * pred_t[i]);		// Second Harmonic
		pred_x[i][4] = cos(4 * M_PI * pred_t[i]);
	}

	// Normalise years:
	norm_f(3, pred_x);

	// Use new adjusted predicted features to predict temperature!
	cout << "\nPredicted Temperatures are: " << endl;
	double *pred_y = new double[3];
	for (int i = 0; i < 3; i++) cout << "Date: " << pred_t[i] << "\tPredicted Avg Temp: " << reg.hypothesis(pred_x[i]) << endl;


	// ------ End of File ------ 
	//Delete all pointers
	delete[] date, test_y, weights, yi, af, pred_af, sh, pred_sh, pred_y;
	delete gnuplotPipe;
	delete_double(samples, xi);
	delete_double(samples, data);
	delete_double(samples, ti);
	delete_double(3, pred_x);

	system("pause");
	return 0;
}
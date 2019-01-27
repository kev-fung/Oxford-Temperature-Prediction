
/* ACSE 5 -
	Langrange Interpolation
*/

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <cmath>
#include <vector>
#include <assert.h>

using namespace std;

void printvector(vector <vector <string> > data)
{

	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].size(); j++)
			cout << data[i][j] << " ";
		cout << data[i].size() << endl;
	}
}


vector <vector <string> > read_data(string fname)
{
	/* Store all data from file into a string 2D vector*/
	vector <vector <string> > data;

	fstream myFile;
	myFile.open(fname, fstream::in);

	if (myFile.fail()) {
		cerr << "File opening failed";
		return data;
	}

	string line;
	while (getline(myFile, line))
	{
		if (line.length() == 0)	continue;
		// Skip descriptive header lines:
		if (line.find("   ") == string::npos) continue;
		// Skip column title lines:
		if (line.find("sun") != string::npos || line.find("hours") != string::npos) continue;
		// Skip "---" lines to remove samples without sun hours:
		if (line.find("---") != string::npos) continue;

		istringstream ss(line);
		vector <string> record;

		string separated_line;

		// Split line by delimiter and record:
		int col_count = 0;
		size_t pos = 0;
		size_t next_pos = 1;
		size_t cpos;
		bool check;
		string subpart;
		string delim = " ";
		char unwanted_char = '*';
		string unwanted_str = "Provisional";

		//line.erase(0, pos + delim.length());	// Erase first tab in our data
		while ((pos = line.find(delim)) != string::npos)
		{
			check = true;
			while (check)
			{
				if (line.find(delim, pos + 1) == pos + 1)
				{
					line.erase(0, pos + delim.length());
					pos = line.find(delim);
				}
				else
				{
					line.erase(0, pos + delim.length());
					check = false;
				}
			}

			pos = line.find(delim);
			subpart = line.substr(0, pos);

			// Remove any unwanted char in subpart:
			cpos = subpart.find(unwanted_char);
			if (cpos != string::npos) subpart.erase(cpos, 1);

			// Remove any specific words i.e. Provisional:
			cpos = subpart.find(unwanted_str);
			if (cpos != string::npos) continue;


			//if (col_count == 1) {
			//	if (subpart != "1") {

			//		break;
			//	}
			//}

			record.push_back(subpart);

			line.erase(0, pos + delim.length());
			col_count++;
		}
		data.push_back(record);
	}

	myFile.close();

	return data;
}


void convert_to_double(vector < vector <string> > stringdata, double **data)
{
	// Convert 2D Vector string to double pointer array
	double val;
	for (int i = 0; i < stringdata.size(); i++)
	{
		// Set each double pointer element to be an array of columnsize
		data[i] = new double[stringdata[i].size()];

		for (int j = 0; j < stringdata[i].size(); j++)
		{
			val = atof(stringdata[i][j].c_str());
			data[i][j] = val;
		}
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


double hypothesis(int numFeatures, double *weights, double *x)
{
	double hyp = 0;

	for (int i = 0; i < numFeatures; i++)
	{
		hyp += x[i] * weights[i];
	}

	return hyp;
}


double cost_function(const int &numFeatures, const int &samples, double *weights, double **x, double *y)
{
	double cost = 0.0;

	for (int i = 0; i < samples; i++)
	{
		cost += pow(hypothesis(numFeatures, weights, x[i]) - y[i], 2.0);
	}

	cost = cost * (1.0 / (2.0 * samples));

	return cost;
}


void gradientDescent(const int &numFeatures, const int &samples, int &iter_max, double *weights, double **x, double *y, double &alpha, double &tol)
{
	int iter = 0;
	bool not_converged = false;
	double old_cost = 1, new_cost = 0, sum2 = 0;
	double k = alpha / samples;

	while (abs(new_cost - old_cost) > tol)
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
				sum2 += (hypothesis(numFeatures, weights, x[i]) - y[i]) * x[i][j];
			}
			weights[j] = weights[j] - k * sum2;
		}

		old_cost = new_cost;
		new_cost = cost_function(numFeatures, samples, weights, x, y);

		if (0 == iter % 200)
		{
			cout << "Working...\t loop: " << iter << "   \t";
			cout << "Error: " << new_cost << endl;
		}

		iter++;
	}
	if (not_converged == false) cout << "Convergence achieved" << endl;

}


void feature_scaling(const int &samples, const int &numFeatures, double **x)
{
	double *mean = new double[numFeatures];
	double *max = new double[numFeatures];
	double *min = new double[numFeatures];

	for (int i = 0; i < numFeatures; i++) mean[i] = 0;
	for (int i = 0; i < numFeatures; i++) max[i] = -1e9;
	for (int i = 0; i < numFeatures; i++) min[i] = 1e9;

	for (int j = 0; j < numFeatures; j++)
	{
		for (int i = 0; i < samples; i++)
		{
			// Find mean - summation term:
			mean[j] += x[i][j];

			// Find max:
			if (x[i][j] > max[j]) {
				max[j] = x[i][j];
			}

			// Find min:
			if (x[i][j] < min[j]) {
				min[j] = x[i][j];
			}
		}
		mean[j] = mean[j] / samples;
	}

	// Feature scaling: mean normalisation of our x data:
	for (int m = 0; m < samples; m++)
	{
		for (int n = 1; n < numFeatures; n++)
		{
			x[m][n] = (x[m][n] - mean[n]) / (max[n] - min[n]);
		}
	}

	delete[] mean, max, min;
}


void mean_norm(const int &samples, double *x)
{
	double mean = 0.0;
	double max = -1e9;
	double min = 1e9;

	for (int i = 0; i < samples; i++)
	{
		// Find mean - summation term:
		mean += x[i];

		// Find max:
		if (x[i] > max) max = x[i];

		// Find min:
		if (x[i] < min) min = x[i];
	}

	mean = mean / samples;

	// mean normalisation:
	for (int m = 0; m < samples; m++)
	{
		x[m] = (x[m] - mean) / (max - min);
	}
}



//void feature_scaling(const int &samples, int columns[], int &numCol, double **x)
//{
//	int j;
//	double *mean = new double[numCol];
//	double *max = new double[numCol];
//	double *min = new double[numCol];
//
//	for (int i = 0; i < numCol; i++) mean[i] = 0;
//	for (int i = 0; i < numCol; i++) max[i] = -1e9;
//	for (int i = 0; i < numCol; i++) min[i] = 1e9;
//
//	for (int k = 0; k < numCol; k++)
//	{
//		j = columns[k];
//		for (int i = 0; i < samples; i++)
//		{
//			// Find mean - summation:
//			mean[k] += x[i][j];
//
//			// Find max:
//			if (x[i][j] > max[k]) {
//				max[k] = x[i][j];
//			}
//
//			// Find min:
//			if (x[i][j] < min[k]) {
//				min[k] = x[i][j];
//			}
//		}
//		mean[k] = mean[k] / samples;
//	}
//
//	// Feature scaling: mean normalisation of our x data:
//	for (int m = 0; m < samples; m++)
//	{
//		for (int n = 0; n < numCol; n++)
//		{
//			j = columns[n];
//			x[m][j] = (x[m][j] - mean[j]) / (max[j] - min[j]);
//		}
//	}
//
//	delete[] mean, max, min;
//}


int main()
{
	string fname = "oxforddata.txt";

	// Read data into a vector:
	vector<vector<string> > stringdata = read_data(fname);

	// Initialise double pointer outside of function:
	// Double pointer now contains array of rowsize elements
	double **data = new double*[stringdata.size()];
	convert_to_double(stringdata, data);

	// Store meta-data:
	const int samples = stringdata.size();
	int num_columns = stringdata[1].size();

	// Clear vector
	stringdata.clear();


	// Pick out data columns that I want from original data:
	double **xi = new double*[samples];
	double *yi = new double[samples];

	// Number of features to include - year, month, afdays, rainfall, sun hours.
	const int numFeatures = 10 + 1;

	// Fill xi and yi:
	for (int i = 0; i < samples; i++)
	{
		double cont_time = data[i][0] + (data[i][1] / 12.0);
		xi[i] = new double[numFeatures];

		// Feature Selection: Modelling our hypothesis/polynomial function:
		xi[i][0] = 1.0;								// Intercept term
		xi[i][5] = data[i][4];						// Airfrost days
		xi[i][6] = pow(data[i][4], 2.0);			// Airfrost days ^2
		xi[i][7] = data[i][6];						// Sun hours
		xi[i][8] = cont_time * data[i][6];			// Interaction - Time x Sunhours
		xi[i][9] = cont_time * data[i][4];			// Interaction - Time x Airfrost 
		xi[i][10] = cont_time * xi[i][6];			// Interaction - Time x Airfrost^2

		//xi[i][-] = data[i][5];					// Rainfall - Disincluded, no correlation in data

		// Continuous date/time modelled as sine wave:
		xi[i][1] = sin(2 * M_PI * cont_time);		// First Harmonic
		xi[i][2] = cos(2 * M_PI * cont_time);
		xi[i][3] = sin(4 * M_PI * cont_time);		// Second Harmonic
		xi[i][4] = cos(4 * M_PI * cont_time);

		// Work out the average temperature:
		yi[i] = (data[i][2] + data[i][3]) / 2;
	}

	// Mean-Normalise Features:
	double *af = new double[samples];
	double *af2 = new double[samples];
	double *sh = new double[samples];
	double *shd = new double[samples];
	double *afd = new double[samples];
	double *afd2 = new double[samples];

	for (int i = 0; i < samples; i++)
	{
		af[i] = xi[i][5];
		af2[i] = xi[i][6];
		sh[i] = xi[i][7];
		shd[i] = xi[i][8];
		afd[i] = xi[i][9];
		afd2[i] = xi[i][10];
	}

	mean_norm(samples, af);
	mean_norm(samples, af2);
	mean_norm(samples, sh);
	mean_norm(samples, shd);
	mean_norm(samples, afd);
	mean_norm(samples, afd2);

	for (int i = 0; i < samples; i++)
	{
		xi[i][5] = af[i];
		xi[i][6] = af2[i];
		xi[i][7] = sh[i];
		xi[i][8] = shd[i];
		xi[i][9] = afd[i];
		xi[i][10] = afd2[i];
	}

	delete[] af, af2, sh, shd, afd, afd2;

	// Output other data for analysis:
	// Commented out for efficiency reasons...
	//
	//double *af = new double[samples];
	//double *rainfall = new double[samples];
	//double *sunhours = new double[samples];
	//for (int i = 0; i < samples; i++)
	//{
	//	af[i] = data[i][4];
	//	rainfall[i] = data[i][5];
	//	sunhours[i] = data[i][6];
	//}
	//fileoutput("af_vs_t1.dat", samples, yi, af);
	//fileoutput("rf_vs_t1.dat", samples, yi, rainfall);
	//fileoutput("sh_vs_t1.dat", samples, yi, sunhours);
	//
	//delete[] af, rainfall, sunhours;


	// Output original data:

	// Assign new date array for plotting:
	double *date = new double[samples];
	for (int i = 0; i < samples; i++) date[i] = (data[i][0]) + (data[i][1] / 12.0);

	fileoutput("input.dat", samples, yi, date);


	// Extrapolate using linear regression:
	double *weights = new double[numFeatures];
	for (int i = 0; i < numFeatures; i++) weights[i] = 0.0;

	// Learning rate
	double alpha = 0.1;
	// Tolerance for convergence
	double abs_tol = 1e-3;
	// Maximum iterations allowed
	int iter_max = 1e7;

	// Train function i.e. compute new weights:
	cout << "Testing our gradient descent: " << endl;
	gradientDescent(numFeatures, samples, iter_max, weights, xi, yi, alpha, abs_tol);
	cout << "\nFound weights for our polnomial: " << endl;
	for (int i = 0; i < numFeatures; i++) cout << weights[i] << ", ";

	// Check our cost i.e. error in the new function:
	double cost = cost_function(numFeatures, samples, weights, xi, yi);
	cout << "\nError of our found function: " << cost << endl;

	// Backtesting our found polynomial:
	// i.e. feed original x data and see what the found y is:
	double *test_y = new double[samples];
	for (int i = 0; i < samples; i++) test_y[i] = hypothesis(numFeatures, weights, xi[i]);
	fileoutput("test.dat", samples, test_y, date);


	// End of code: Delete all pointers
	delete[] date, test_y, weights, yi;
	delete_double(samples, xi);
	delete_double(samples, data);

	system("pause");
	return 0;
}
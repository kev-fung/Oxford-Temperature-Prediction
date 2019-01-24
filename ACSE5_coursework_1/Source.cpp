
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
	double k;
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
	double old_cost = 0, new_cost = 9999, sum2 = 0;
	double k = alpha / samples;

	//for (int it = 0; it < iter_max; it++ )
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

		if (0 == iter % 50)
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
			// Find mean - summation:
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

	// (IF HAVE TIME, CHANGE THIS SO IT READS INTO MEMORY INSTEAD):
	// Read data into a vector:
	vector<vector<string> > stringdata = read_data(fname);
	//printvector(stringdata);
	//system("pause");

	// Initialise double pointer outside of function:
	// Double pointer now contains array of rowsize elements
	double **data = new double*[stringdata.size()];
	convert_to_double(stringdata, data);

	const int samples = stringdata.size();
	int num_columns = stringdata[1].size();
	//print_double(samples, num_columns, data);
	//system("pause");

	// Clear vector memory
	stringdata.clear();

	//// Find number of jan samples:
	//double jan_samples = 0;
	//for (int i = 0; i < samples; i++)
	//{
	//	if (data[i][1] == 1) jan_samples++;
	//}

	//cout << "Number of January Samples: " << jan_samples << endl;
	double jan_samples = samples;

	// Pick out data columns that I want from original data:
	double **xi = new double*[jan_samples];
	double *yi = new double[jan_samples];

	// Number of features to include - year, month, afdays, rainfall, sun hours
	const int numFeatures = 4 + 1;

	// Fill the xi and yi pointer arrays:
	int j = 0;
	for (int i = 0; i < samples; i++)
	{
		//// Select only january data:
		//if (data[i][1] != 1) continue;

		xi[j] = new double[numFeatures];

		// Modelling our hypothesis/polynomial function:
		xi[j][0] = 1;						// Intercept term
		//xi[j][1] = data[i][4];			// Airfrost days
		//xi[j][2] = data[i][5];			// Rainfall
		//xi[j][3] = data[i][6];			// Sun hours

		// Continuous date/time modelled as sine wave:
		xi[j][1] = sin(2 * M_PI * ((data[i][0]) + (data[i][1] / 12.0)));		// First Harmonic
		xi[j][2] = cos(2 * M_PI * ((data[i][0]) + (data[i][1] / 12.0)));
		xi[j][3] = sin(4 * M_PI * ((data[i][0]) + (data[i][1] / 12.0)));		// Second Harmonic
		xi[j][4] = cos(4 * M_PI * ((data[i][0]) + (data[i][1] / 12.0)));

		// Work out the average temperature:
		yi[j] = (data[i][2] + data[i][3]) / 2;
		j++;
	}
	//print_double(jan_samples, numFeatures, xi);
	//system("pause");

	// Mean normalise specific columns of data:
	// Select columns:
	//int columns[3] = {1,2,3};
	//int numCol = 3;
	//feature_scaling(jan_samples, columns, numCol, xi);
	//feature_scaling(jan_samples, numFeatures, xi);

	//print_double(jan_samples, numFeatures, xi);
	//system("pause");


	// Output original data:
	double *date = new double[jan_samples];
	j = 0;

	for (int i = 0; i < samples; i++)
	{
		// Output only january dates
		//if (data[i][1] != 1) continue;
		date[j] = (data[i][0]) + (data[i][1] / 12.0);
		//date[j] = data[i][0];
		j++;
	}

	fileoutput("input.dat", jan_samples, yi, date);

	//print_double(jan_samples, numFeatures, xi);
	//system("pause");


	// Extrapolate using linear regression:
	double *weights = new double[numFeatures];
	for (int i = 0; i < numFeatures; i++) weights[i] = 0.0;

	// Learning rate
	double alpha = 0.001;
	// Tolerance for convergence
	double abs_tol = 1e-3;
	// Maximum iterations allowed
	int iter_max = 10000;

	// Train function i.e. compute new weights:
	cout << "Testing our gradient descent: " << endl;
	gradientDescent(numFeatures, jan_samples, iter_max, weights, xi, yi, alpha, abs_tol);
	cout << "\nFound weights for our polnomial: " << endl;
	for (int i = 0; i < numFeatures; i++) cout << weights[i] << ", ";

	// Check our cost i.e. error in the new function:
	double cost = cost_function(numFeatures, jan_samples, weights, xi, yi);
	cout << "\nError of our found function: " << cost << endl;

	// Plot our function: 
	// i.e. feed original x data and see what the found y is:
	double *test_y = new double[jan_samples];

	for (int i = 0; i < jan_samples; i++)
	{
		test_y[i] = hypothesis(numFeatures, weights, xi[i]);
	}

	// Output test data:
	fileoutput("test.dat", jan_samples, test_y, date);

	delete[] date, test_y, weights, yi;
	delete_double(jan_samples, xi);
	delete_double(samples, data);

	system("pause");
	return 0;
}
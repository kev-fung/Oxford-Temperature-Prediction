#define _USE_MATH_DEFINES
#include "Data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <math.h>


using namespace std;

Data::Data(std::string name, int numfeat) : fname(name), numFeatures(numfeat)
{
}


Data::~Data()
{
	for (int i = 0; i < drows; i++)
	{
		delete[] data[i], xi[i], ti[i];
	}
	delete[] data, xi, yi, af, sh, ti;
}


Data::Data(const Data &d) : data(new double*[d.drows]),
xi(new double*[d.drows]),
ti(new double*[d.drows]),
af(new double[d.drows]),
sh(new double[d.drows]),
yi(new double[d.drows]),
fname(d.fname),
numFeatures(d.numFeatures),
drows(d.drows),
dcols(d.dcols)
{
	for (int k = 0; k < drows; k++)
	{
		yi[k] = d.yi[k];
		data[k] = new double[dcols];
		xi[k] = new double[numFeatures];
		ti[k] = new double[numFeatures];
		af[k] = d.af[k];
		sh[k] = d.sh[k];

		for (int j = 0; j < dcols; j++)
		{
			data[k][j] = d.data[k][j];
		}
		for (int j = 0; j < numFeatures; j++)
		{
			xi[k][j] = d.xi[k][j];
			ti[k][j] = d.ti[k][j];
		}
	}
}


void Data::Read_data()
{
	fstream myFile;
	myFile.open(fname, fstream::in);

	if (myFile.fail()) {
		cerr << "File opening failed";
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

			record.push_back(subpart);

			line.erase(0, pos + delim.length());
			col_count++;
		}
		vectordata.push_back(record);
	}

	myFile.close();
}


void Data::Convert_data()
{
	// Convert 2D Vector string to double pointer array
	double val;
	drows = vectordata.size();
	dcols = vectordata[1].size();
	data = new double*[drows];

	for (int i = 0; i < drows; i++)
	{
		// Set each double pointer element to be an array of columnsize
		data[i] = new double[vectordata[i].size()];

		for (int j = 0; j < vectordata[i].size(); j++)
		{
			val = atof(vectordata[i][j].c_str());
			data[i][j] = val;
		}
	}

	// Free up vector
	vectordata.clear();
}


void Data::Select_features()
{
	xi = new double*[drows];

	// Fill xi and yi:
	for (int i = 0; i < drows; i++)
	{
		double cont_time = data[i][0] + (data[i][1] / 12.0);
		xi[i] = new double[numFeatures];

		// Modelling our hypothesis/polynomial function:
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

	}

}


void Data::Select_only_time()
{
	ti = new double*[drows];

	for (int i = 0; i < drows; i++)
	{
		ti[i] = new double[2];

		// Modelling our hypothesis/polynomial function:
		ti[i][0] = 1.0;								// Intercept term
		ti[i][1] = data[i][0] + (data[i][1] / 12.0);	// Linear time
	}
}


void Data::Select_only_af()
{
	af = new double[drows];

	for (int i = 0; i < drows; i++)	af[i] = data[i][4];
}


void Data::Select_only_sh()
{
	sh = new double[drows];

	for (int i = 0; i < drows; i++) sh[i] = data[i][6];
}



void Data::Select_temp()
{
	yi = new double[drows];

	for (int i = 0; i < drows; i++)
	{
		// Work out the average temperature:
		yi[i] = (data[i][2] + data[i][3]) / 2;
	}

}


void Data::Norm_features()
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

	Mean_norm(drows, af);
	Mean_norm(drows, af2);
	Mean_norm(drows, sh);
	Mean_norm(drows, shd);
	Mean_norm(drows, afd);
	Mean_norm(drows, afd2);

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


void Data::Mean_norm(const int &samples, double *x)
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


int Data::Get_nsamples()
{
	return drows;
}

double Data::Get_dcols()
{
	return dcols;
}


double * Data::Get_y()
{
	double * y = new double[drows];
	for (int i = 0; i < drows; i++)
	{
		y[i] = yi[i];
	}
	return y;
}


double ** Data::Get_x()
{
	double ** x = new double*[drows];
	for (int i = 0; i < drows; i++)
	{
		x[i] = new double[numFeatures];
		for (int j = 0; j < numFeatures; j++)
		{
			x[i][j] = xi[i][j];
		}
	}
	return x;
}

double ** Data::Get_data()
{

	double ** d = new double*[drows];
	for (int i = 0; i < drows; i++)
	{
		d[i] = new double[dcols];
		for (int j = 0; j < dcols; j++)
		{
			d[i][j] = data[i][j];
		}
	}
	return d;
}


double * Data::Get_af()
{
	double * x = new double[drows];
	for (int i = 0; i < drows; i++) x[i] = af[i];
	return x;
}


double * Data::Get_sh()
{
	double * x = new double[drows];
	for (int i = 0; i < drows; i++) x[i] = sh[i];
	return x;
}


double ** Data::Get_ti()
{
	double ** x = new double*[drows];
	for (int i = 0; i < drows; i++)
	{
		x[i] = new double[numFeatures];
		for (int j = 0; j < numFeatures; j++)
		{
			x[i][j] = ti[i][j];
		}
	}
	return x;
}
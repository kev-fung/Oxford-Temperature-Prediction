#pragma once
#include <string>
#include <vector>
class Data
{
public:
	Data(std::string name, int numfeat);
	~Data();
	Data(const Data &d);

	void Read_data();
	void Convert_data();
	void Select_features();
	void Select_temp();
	void Norm_features();

	int Get_nsamples();
	double Get_dcols();
	double *Get_y();
	double *Get_af();
	double *Get_sh();
	double **Get_x();
	double **Get_ti();
	double **Get_data();

	void Select_only_time();
	void Select_only_af();
	void Select_only_sh();

private:
	const std::string fname;
	const int numFeatures;
	int drows;
	int dcols;
	double *yi;
	double *af;
	double *sh;
	double **ti;
	double **xi;
	double **data;

	void Mean_norm(const int &samples, double *x);

	// Temporary vector storage:
	std::vector<std::vector<std::string> > vectordata;

};


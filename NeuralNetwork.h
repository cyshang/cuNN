#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include <Eigen/Core>

struct NeuralNetwork
{
	// Network info
	int nLayer;
	std::vector<int> nNeuron;

	// Training parameters
	double train_ratio;
	double init_mu;
	int fit_times;
	int max_epoch;

	// Data
	int nSample;
	int tSample;
	int vSample;
	Eigen::MatrixXd inputX;
	Eigen::RowVectorXd target_val;
	double max_val;
	double min_val;
	double avg_val;
}; 

#endif


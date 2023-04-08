
// This code trains a Support Vector Regression (SVR) model and uses it to predict the output for a new input. 
// The SVR model is trained using a Gaussian kernel and the alpha values for each data point are 
// computed using the kernel function. The weight vector and bias term are then computed using the 
// alpha values and the training data. The trained model is then used to predict the output for a new 
// input by computing the dot product of the weight vector and the input vector, and adding the bias term. 
// The code also includes functions for computing the dot product of two vectors and the Gaussian kernel 
// between two vectors. The sample data used to train and test the model consists of five two-dimensional 
// feature vectors and their corresponding output values.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/// Data point
struct Data_Node {
	/// Feature vector
	double* x;

	/// Output value
	double y;
};

/// Trained model
struct Model {
	/// Alpha values
	double* alphas;

	/// Bias term
	double b;
};

/// Calculate the dot product of two vectors of size `n`
static double dot_product(double* a, double* b, int n) {
	double result = 0;
	for (int i = 0; i < n; ++i) {
		result += a[i] * b[i];
	}

	return result;
}

/// Calculate the Gaussian kernel between two vectors of size `n`
static double kernel(double* x1, double* x2, int n, double sigma) {
	double result = 0;
	for (int i = 0; i < n; ++i) {
		// Calculate squared difference between features
		double xi1 = x1[i];
		double xi2 = x2[i];
		result += pow(xi1 - xi2, 2);
	}

	// Calculate kernel using Gaussian function
	return exp(-result / (2 * pow(sigma, 2)));
}

/// Compute alpha values for each data point
static double* get_alphas(struct Data_Node* data, int n, int m, double sigma) {
	// Initialize alpha values to zero
	double* result = (double*)malloc(n * sizeof(double));
	for (int i = 0; i < n; ++i) {
		result[i] = 0;
	}

	double alpha_sum = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			alpha_sum += result[j] * data[j].y * kernel(data[i].x, data[j].x, m, sigma);
		}

		double denominator = kernel(data[i].x, data[i].x, m, sigma);
		if (denominator == 0) {
			// Avoid division by zero
			result[i] = 0;
		}
		else {
			// Update alpha values for this data point
			result[i] = (1 - data[i].y * alpha_sum) / kernel(data[i].x, data[i].x, m, sigma);
		}

		alpha_sum = 0;
	}

	return result;
}

/// Train an SVR model using the given data
static struct Model train_svr(struct Data_Node* data, int n, int m, double sigma) {
	// Compute alpha values for the given data
	double* alphas = get_alphas(data, n, m, sigma);

	// Compute bias term using the alpha values and the training data
	double b_sum = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			b_sum += alphas[j] * data[j].y * kernel(data[i].x, data[j].x, m, sigma);
		}
		b_sum = data[i].y - b_sum;
	}
	double b = b_sum / n;

	// Create the trained model
	struct Model result = { alphas, b };

	return result;
}

/// Use the trained model to predict the output for new input
static double predict_output(struct Model model, double* x, struct Data_Node* data, int n, int m, double sigma) {
	double result = 0;
	for (int i = 0; i < n; ++i) {
		result += model.alphas[i] * data[i].y * kernel(data[i].x, x, m, sigma);
	}
	result += model.b;
	return result;
}


#include <stdio.h>
#include <stdlib.h>
#include "svm.h"

struct svm_model* libsvm_train_svr_model(struct svm_node x[], double y[], int length) {
	struct svm_problem problem = {
		.x = x,
		.y = y,
		.l = length,
	};
	struct svm_parameter param = {
		.svm_type = EPSILON_SVR,
		.kernel_type = RBF,
		.degree = 3,
		.gamma = 0,
		.coef0 = 0,
		.nu = 0.5,
		.cache_size = 100,
		.C = 1,
		.eps = 1e-3,
		.p = 0.1,
		.shrinking = 1,
		.probability = 0,
		.nr_weight = 0,
		.weight_label = NULL,
		.weight = NULL,
	};
   
    return svm_train(&problem, &param);
}

double libsvm_predict_svr_output(struct svm_model* model, double x[]) {
    struct svm_node input[] = { {1, x[0]}, {2, x[1]}, {-1, -1} };
    double prediction = svm_predict(model, input);
    return prediction;
}



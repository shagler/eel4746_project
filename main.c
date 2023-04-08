#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "svm.h"
#define num_features 4
int main(int argc, char** argv) {
    // Load data
    FILE* fp = fopen("META_2012.csv", "r");
    if (fp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    char line[1024];
    fgets(line, 1024, fp); // Skip header line
    int num_samples = 0;
    while (fgets(line, 1024, fp)) {
        num_samples++;
    }
    rewind(fp);
    fgets(line, 1024, fp); // Skip header line

    double* adj_close = malloc(num_samples * sizeof(double));
    double* high = malloc(num_samples * sizeof(double));
    double* low = malloc(num_samples * sizeof(double));
    double* open = malloc(num_samples * sizeof(double));
    double* volume = malloc(num_samples * sizeof(double));

    int i = 0;
    while (fgets(line, 1024, fp)) {
        char* tok = strtok(line, ",");
        int col = 0;
        while (tok != NULL) {
            if (col == 1) {
                adj_close[i] = atof(tok);
            }
            else if (col == 2) {
                high[i] = atof(tok);
            }
            else if (col == 3) {
                low[i] = atof(tok);
            }
            else if (col == 4) {
                open[i] = atof(tok);
            }
            else if (col == 5) {
                volume[i] = atof(tok);
            }
            col++;
            tok = strtok(NULL, ",");
        }
        i++;
    }
    fclose(fp);

    // Preprocess data
    int forecast_percent = 10;
    int forecast_ceil = ceil(forecast_percent * num_samples / 100.0);
    double* high_low_per = malloc(num_samples * sizeof(double));
    double* per_change = malloc(num_samples * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        high_low_per[i] = (high[i] - adj_close[i]) / adj_close[i] * 100;
        per_change[i] = (open[i] - open[i]) / adj_close[i] * 100;
    }
    double* features[] = { adj_close, high_low_per, per_change, volume };
    double* X = malloc(num_samples * num_features * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            X[i * num_features + j] = features[j][i];
        }
    }
    double* y = malloc(num_samples * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        if (i + forecast_ceil < num_samples) {
            y[i] = adj_close[i + forecast_ceil];
        }
        else {
            y[i] = NAN;
        }
    }

    // Split data into training and testing sets
    double* X_train = malloc((num_samples - forecast_ceil) * num_features * sizeof(double));
    double* X_test = malloc(forecast_ceil * num_features * sizeof(double));
    double* y_train = malloc(num_samples - forecast_ceil * sizeof(double));
    double* y_test = malloc(forecast_ceil * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            X_train[i * num_features + j] = X[i * num_features + j];
        }
        y_train[i] = y[i];
    }
    for (int i = 0; i < forecast_ceil; i++) {
        for (int j = 0; j < num_features; j++) {
            X_test[i * num_features + j] = X[(num_samples - forecast_ceil + i) * num_features + j];
        }
        y_test[i] = y[num_samples - forecast_ceil + i];
    }

    // Train model
    struct svm_problem problem = {
        .x = X_train,
        .y = y_train,
        .l = num_samples - forecast_ceil,
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
    struct svm_model* model = svm_train(&problem, &param);

    // Test model
    double* forecast_set = malloc(forecast_ceil * sizeof(double));
    for (int i = 0; i < forecast_ceil; i++) {
        double x[num_features];
        for (int j = 0; j < num_features; j++) {
            x[j] = X_test[i * num_features + j];
        }
        forecast_set[i] = svm_predict(model, x);
    }

    // Print accuracy
    double accuracy = svm_predict(model, X_test, y_test, forecast_ceil);
    printf("accuracy: %lf\n", accuracy);

    // Add forecast to dataframe
    for (int i = 0; i < forecast_ceil; i++) {
        y[num_samples - forecast_ceil + i] = forecast_set[i];
    }

    // Plot results
    printf("Date,Adj Close,Forecast\n");
    for (int i = 0; i < num_samples; i++) {
        printf("%d,%lf,%lf\n", i, adj_close[i], y[i]);
    }

    // Cleanup
    svm_free_and_destroy_model(&model);
    free(adj_close);
    free(high);
    free(low);
    free(open);
    free(volume);
    free(high_low_per);
    free(per_change);
    free(X);
    free(y);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    free(forecast_set);

    return 0;
}
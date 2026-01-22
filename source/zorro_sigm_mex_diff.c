#include "mex.h"
#include <math.h>

void kernel_double(double *Y, const double *X, mwSize N,
                   double slope, double alpha, double k){
    double km1 = k - 1.0;
    for (mwSize i = 0; i < N; i++) {
        double x = slope * X[i] + 0.5;
        if (x < 0.0) {
            double G = 1.0 / (1.0 + km1 * exp(-alpha * x));
            Y[i] = k * G * (1.0 + alpha * x * (1.0 - G)) * slope;
        }
        else if (x > 1.0) {
            double t = 1.0 - x;
            double G = 1.0 / (1.0 + km1 * exp(-alpha * t));
            Y[i] = k * G * (1.0 + alpha * t * (1.0 - G)) * slope;
        }
        else {
            Y[i] = slope;
        }
    }
}

void kernel_float(float *Y, const float *X, mwSize N,
                  float slope, float alpha, float k){
    float km1 = k - 1.0f;
    for (mwSize i = 0; i < N; i++) {
        float x = slope * X[i] + 0.5f;
        if (x < 0.0f) {
            float G = 1.0f / (1.0f + km1 * expf(-alpha * x));
            Y[i] = k * G * (1.0f + alpha * x * (1.0f - G)) * slope;
        }
        else if (x > 1.0f) {
            float t = 1.0f - x;
            float G = 1.0f / (1.0f + km1 * expf(-alpha * t));
            Y[i] = k * G * (1.0f + alpha * t * (1.0f - G)) * slope;
        }
        else {
            Y[i] = slope;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    mwSize N = mxGetNumberOfElements(prhs[0]);
    
    // input parameters vector
    double *p = mxGetPr(prhs[1]);

    if (mxIsDouble(prhs[0])) {
        // input matrix
        const double *X = mxGetPr(prhs[0]);
        // output matrix
        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxREAL);
        double *Y = mxGetPr(plhs[0]);
        // process
        kernel_double(Y, X, N, p[0], p[1], p[2]);
        return;
    }
    if (mxIsSingle(prhs[0])) {
        // input matrix
        const float *X = (const float *) mxGetPr(prhs[0]);
        plhs[0] = mxCreateNumericMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxSINGLE_CLASS, mxREAL);
        float *Y = (float *) mxGetPr(plhs[0]);
        // process
        kernel_float(Y, X, N, (float)p[0], (float)p[1], (float)p[2]);
        return;
    }
}
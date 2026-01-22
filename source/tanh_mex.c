#include "mex.h"
#include <math.h>

void kernel_double(double *Y, const double *X, mwSize N){
    for (mwSize i = 0; i < N; i++) {
        Y[i] = tanh(X[i]);
    }
}

void kernel_float(float *Y, const float *X, mwSize N){
    for (mwSize i = 0; i < N; i++) {
        Y[i] = tanhf(X[i]);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    mwSize N = mxGetNumberOfElements(prhs[0]);

    if (mxIsDouble(prhs[0])) {
        // input matrix
        const double *X = mxGetPr(prhs[0]);
        // output matrix
        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxREAL);
        double *Y = mxGetPr(plhs[0]);
        // process
        kernel_double(Y, X, N);
        return;
    }
    if (mxIsSingle(prhs[0])) {
        // input matrix
        const float *X = (const float *) mxGetPr(prhs[0]);
        plhs[0] = mxCreateNumericMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxSINGLE_CLASS, mxREAL);
        float *Y = (float *) mxGetPr(plhs[0]);
        // process
        kernel_float(Y, X, N);
        return;
    }
}
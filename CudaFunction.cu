__global__ void     // calculate A = f(Z + b) in neural network, Z = W * A here
CalcHidden (int nRow, int nCol, double *Z, int ldZ, const double *b, int incb)
{    
    int iRow = threadIdx.x;
    int pos = blockIdx.x * ldZ + threadIdx.x;
    
    if (iRow < nRow) {
        double tmp = Z[pos] + b[iRow * incb];

        Z[pos] = tmp / sqrt(tmp * tmp + 1);
    }

    return;
}
//            -------- blockIdx --------
//            |                         |
//            | threadIdx               |
//            |                         |
//            --------------------------
// =================================================================================

__global__ void     // calculate A = f(Z + b) in neural network, Z = W * A here
CalcOutput (int nCol, double *dZ, int incZ, const double *b)
{    
    int iCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (iCol < nCol) {
        int pos = iCol * incZ;

        Z[pos] += b;
    }

    return;
}

__global__ void
Calc_dZ (int nRow, int nCol, double *Z, int ldZ, const double *dZ)
{

}
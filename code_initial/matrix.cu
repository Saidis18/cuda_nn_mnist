#include "matrix.h"
#include "error.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res;
    res = (matrix_t *)malloc(sizeof(matrix_t));
    CHECK_ERROR(cudaMalloc(&(res->m), columns * rows * sizeof(double)));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    CHECK_ERROR(cudaFree(m->m));
    free(m);
}

void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    double *h_m = (double *)malloc(m->columns * m->rows * sizeof(double));
    CHECK_ERROR(cudaMemcpy(h_m, m->m, m->columns * m->rows * sizeof(double), cudaMemcpyDeviceToHost));

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row++)
    {
        for (int col = 0; col < lim_col; col++)
        {
            printf("%.2lf ", h_m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns)
            printf("...");
        printf("\n");
    }
    free(h_m);
    if (is_short && lim_rows != m->rows)
        printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    hadamard_product_kernel<<<grid_size, block_size>>>(m1->m, m2->m, res->m, size);
}

__global__ void hadamard_product_kernel(double *m1, double *m2, double *res, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = m1[idx] * m2[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    matrix_sum_kernel<<<grid_size, block_size>>>(m1->m, m2->m, res->m, size);
}

__global__ void matrix_sum_kernel(double *m1, double *m2, double *res, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = m1[idx] + m2[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    matrix_minus_kernel<<<grid_size, block_size>>>(m1->m, m2->m, res->m, size);
}

__global__ void matrix_minus_kernel(double *m1, double *m2, double *res, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = m1[idx] - m2[idx];
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    dim3 block_size(16, 16);
    dim3 grid_size((res->columns + block_size.x - 1) / block_size.x, (res->rows + block_size.y - 1) / block_size.y);
    matrix_dot_kernel<<<grid_size, block_size>>>(m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
}

__global__ void matrix_dot_kernel(double *m1, double *m2, double *res, unsigned m1_rows, unsigned m1_cols, unsigned m2_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1_rows && col < m2_cols)
    {
        int idx = col + row * m2_cols;
        double var = 0.0;

        for (int ii = 0; ii < m1_cols; ii++)
        {
            var += m1[ii + row * m1_cols] * m2[col + ii * m2_cols];
        }

        res[idx] = var;
    }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert((m1->columns == res->columns) &&
           (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    matrix_function_kernel<<<grid_size, block_size>>>(m1->m, f, res->m, size);
}

__global__ void matrix_function_kernel(double *m, double (*f)(double), double *res, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = f(m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert((m1->columns == res->rows) &&
           (m1->rows == res->columns));

    dim3 block_size(16, 16);
    dim3 grid_size((res->columns + block_size.x - 1) / block_size.x, (res->rows + block_size.y - 1) / block_size.y);
    matrix_transpose_kernel<<<grid_size, block_size>>>(m1->m, res->m, m1->rows, m1->columns);
}

__global__ void matrix_transpose_kernel(double *m, double *res, unsigned rows, unsigned cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        res[row + col * rows] = m[col + row * cols];
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert((m1->rows == res->rows) &&
           (m1->columns == res->columns));

    int size = m1->rows * m1->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    matrix_scalar_kernel<<<grid_size, block_size>>>(m1->m, s, res->m, size);
}

__global__ void matrix_scalar_kernel(double *m, double s, double *res, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        res[idx] = m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    CHECK_ERROR(cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(double), cudaMemcpyDeviceToDevice));
}

void ones(matrix_t *m)
{
    int size = m->rows * m->columns;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    ones_kernel<<<grid_size, block_size>>>(m->m, size);
}

__global__ void ones_kernel(double *m, unsigned size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)    {
        m[idx] = 1.0;
    }
}

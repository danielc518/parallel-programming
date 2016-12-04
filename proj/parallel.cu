#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

/* CUDA implementation */

#define n 512 // number of dimensions for input vector

using namespace std;

// [HOST] Computes inner product between vectors 'a' and 'b'
float innerProduct(float* a, float* b, int idx)
{
	float sum = 0.0;

	for (int i = 0; i < n; i++) 
	{
		sum += (a[idx*n+i] * b[idx*n+i]);
	}

	return sum;
}

// [DEVICE] Computes a chunk of summation
__global__ void aggregate(float* u, float* mu, int chunk, int m, float* r)
{
	memset(r+blockIdx.x*n, 0.0, n*sizeof(float));

	// Determine which portion of the summation this block will compute
	int begin = blockIdx.x*chunk;
	int end = min(m, (blockIdx.x+1)*chunk);

	// Compute summation for this block's portion
	for (int i = begin; i < end; i++)
	{
		for (int j = 0; j < n; j++)
		{
			r[blockIdx.x*n+j] += (mu[i] * u[i*n+j]);
		}
	}
}

// [DEVICE] Aggregate partial summation results from each block and finalize vector 'u'
__global__ void finalize(float* u, float* r, int m, int numBlocks)
{
	for (int i = 0; i < numBlocks; i++)
	{
		for (int j = 0; j < n; j++)
		{
			u[m*n+j] -= r[i*n+j];
		}
	}
}

// [DEVICE] Computes the value of mu by parallelizing inner products
__global__ void computeMu(float* v, float* u, float* d_mu, int m)
{
	// Compute inner products

	int u_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int v_idx = threadIdx.x + m * blockDim.x;

	__shared__ float n_temp[n]; // Temporary variable for numerator
	__shared__ float d_temp[n]; // Temporary variable for denominator

	n_temp[threadIdx.x] = u[u_idx] * v[v_idx];
	d_temp[threadIdx.x] = u[u_idx] * u[u_idx];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float n_sum = 0.0;
		float d_sum = 0.0;

		for (int i = 0; i < n; i++)
		{
			n_sum += n_temp[i];
			d_sum += d_temp[i];
		}

		d_mu[blockIdx.x] = n_sum / d_sum; // mu value
	}
}

// Applies Gram-Schmidt process to input matrix (row vectors) 'v' and outputs results to 'u'
void applyGramSchmidt(float* v, float* u, int maxBlocks) 
{
	// Device memory pointers for vectors and intermediate results
	float *d_v, *d_u, *d_mu, *d_r;

	const int matrixSize = n * n * sizeof(float); // Input/output matrix size
	const int tempResSize = n * maxBlocks; // Intermediate results size

	float* r = new float[tempResSize]; // Array to hold intermediate results

	// Allocate device memory
	cudaMalloc(&d_v, matrixSize);
	cudaMalloc(&d_u, matrixSize);
	cudaMalloc(&d_mu, n*sizeof(float));
	cudaMalloc(&d_r, tempResSize*sizeof(float));

	// Copy values from host to device
	cudaMemcpy(d_v, v, matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, d_v, n*sizeof(float), cudaMemcpyDeviceToDevice); // Set vector u_0 to v_0

	// Begin iterations
	for (int m = 1; m < n; m++)
	{
		// Set vector u_m to v_m
		cudaMemcpy(d_u+m*n, d_v+m*n, n*sizeof(float), cudaMemcpyDeviceToDevice);

		// Compute mu by parallelizing inner products
		computeMu<<<m,n>>>(d_v, d_u, d_mu, m);

		// Determine how many blocks to use based on user input and 'm' 
		int numBlocks = m < maxBlocks ? m : maxBlocks;

		// Divide work into chunks of equal sizes
		int chunk = (int)ceil((float)m/(float)numBlocks);

		// Compute a chunk of summation in each block
		aggregate<<<numBlocks,1>>>(d_u, d_mu, chunk, m, d_r);

		// Aggregate partial summation results from each block and finalize vector 'u'
		finalize<<<1,1>>>(d_u, d_r, m, numBlocks);
	}

	// Copy results from device to host
	cudaMemcpy(u, d_u, matrixSize, cudaMemcpyDeviceToHost);

	// Deallocate memory in device
	cudaFree(d_v); cudaFree(d_u); cudaFree(d_mu); cudaFree(d_r);

	delete[] r;

	// Perform normalization
	for (int m = 0; m < n; m++)
	{
		float norm = sqrt(innerProduct(u, u, m));
		for (int j = 0; j < n; j++)
		{
			u[m*n+j] /= norm;
		}
	}
}

// Prints out the results of given matrix
void print(float* output)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << output[i*n+j];
			cout << " ";
		}
		cout << endl;
	}
}

int main(int argc, char *argv[])
{
	// Set seed for random number generation
	srand(1);

	// Get maximum number of blocks to use from command line
	const int maxBlocks = atoi(argv[1]);

	// Initialize matrices (collection of row vectors) 'v' and 'u'
	float* v = new float[n*n];
	float* u = new float[n*n];

	// Fill input matrix 'v' with random numbers
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			v[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}

	// Record start time
	auto startTime = chrono::high_resolution_clock::now();

	// Begin Gram-Schmidt process
	applyGramSchmidt(v, u, maxBlocks);

	// Measure total elapsed time (in seconds)
	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	long duration = chrono::duration_cast<chrono::microseconds>(time).count();
	float sec = (float) duration / 1000000.0;

	// Output time
	cout << sec << endl;

	// Print output matrix
	// print(u);

	// Deallocate memory
	delete[] u;
	delete[] v;

}
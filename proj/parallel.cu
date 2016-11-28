#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define n 3 // number of dimensions
#define blocks 3 // number of blocks

using namespace std;

float innerProduct(float* a, float* b, int idx)
{
	float sum = 0.0;

	for (int i = 0; i < n; i++) 
	{
		sum += (a[idx*n+i] * b[idx*n+i]);
	}

	return sum;
}

__global__ void aggregate(float* u, float* mu, int chunk, int m, float* r)
{
	memset(r+blockIdx.x*n, 0.0, n*sizeof(float));

	int begin = blockIdx.x*chunk;
	int end = min(m, (blockIdx.x+1)*chunk);

	for (int i = begin; i < end; i++)
	{
		for (int j = 0; j < n; j++)
		{
			r[blockIdx.x*n+j] += (mu[i] * u[i*n+j]);
		}
	}
}

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

void applyGramSchmidt(float* v, float* u) 
{
	float *d_v, *d_u, *d_mu, *d_r;

	const int matrixSize = n * n * sizeof(float);
	const int tempResSize = n * blocks;

	float* r = new float[tempResSize]; // Temporary space to hold results

	cudaMalloc(&d_v, matrixSize);
	cudaMalloc(&d_u, matrixSize);
	cudaMalloc(&d_mu, n*sizeof(float));
	cudaMalloc(&d_r, tempResSize*sizeof(float));

	cudaMemcpy(d_v, v, matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, d_v, n*sizeof(float), cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize();

	for (int m = 1; m < n; m++)
	{
		cudaMemcpy(d_u+m*n, d_v+m*n, n*sizeof(float), cudaMemcpyDeviceToDevice);

		computeMu<<<m,n>>>(d_v, d_u, d_mu, m);

		cudaDeviceSynchronize();

		int numBlocks = m < blocks ? m : blocks;

		int chunk = (int)ceil((float)m/(float)numBlocks);

		aggregate<<<numBlocks,1>>>(d_u, d_mu, chunk, m, d_r);

		cudaDeviceSynchronize();

		finalize<<<1,1>>>(d_u, d_r, m, numBlocks);

		cudaDeviceSynchronize();
	}

	cudaMemcpy(u, d_u, matrixSize, cudaMemcpyDeviceToHost);

	cudaFree(d_v); cudaFree(d_u); cudaFree(d_mu); cudaFree(d_r);

	delete[] r;

	for (int m = 0; m < n; m++)
	{
		float norm = sqrt(innerProduct(u, u, m));
		for (int j = 0; j < n; j++)
		{
			u[m*n+j] /= norm;
		}
	}
}

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
	srand(1);

	float* v = new float[n*n];
	float* u = new float[n*n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			v[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}

	auto startTime = chrono::high_resolution_clock::now();

	applyGramSchmidt(v, u);

	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	long duration = chrono::duration_cast<chrono::microseconds>(time).count();
	float sec = (float) duration / 1000000.0;

	cout << sec << endl;

	print(u);

	delete[] u;
	delete[] v;

}
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <omp.h>

/* Parallel (optimized) version of C++ code based on OpenMP */

using namespace std;

// Computes inner product between vectors 'a' and 'b' with loop unrolling 
float innerProduct(float* a, float* b, const int n)
{
	float sum1 = 0.0;
	float sum2 = 0.0;
	float sum3 = 0.0;
	float sum4 = 0.0;
	float sum5 = 0.0;

	for (int i = 0; i < n; i+=5) 
	{
		sum1 += (a[i] * b[i]);
		sum2 += (a[i+1] * b[i+1]);
		sum3 += (a[i+2] * b[i+2]);
		sum4 += (a[i+3] * b[i+3]);
		sum5 += (a[i+4] * b[i+4]);
	}

	return sum1 + sum2 + sum3 + sum4 + sum5;
}

// Applies Gram-Schmidt process to input matrix (row vectors) 'v' and outputs results to 'u'
void applyGramSchmidt(float** v, float** u, const int n, const int maxThreads) 
{
	// Cache to hold inner product computation results
	float* cache = new float[n];

	// Set vector u_0 to v_0
	copy(v[0], v[0]+n, u[0]);

	// Begin iterations
	for (int m = 1; m < n; m++)
	{
		// Set vector u_m to v_m
		copy(v[m], v[m]+n, u[m]);

		// Determine how many threads to use based on user input and 'm' (i.e. number of iterations)
		const int numThreads = m < maxThreads ? m : maxThreads;

		// Divide work into chunks of equal sizes
		const int chunkSize = (int)ceil((float)m/(float)numThreads);

		// 2D array for holding intermediate results from different threads
		float** results = new float*[numThreads];

		// Set number of threads to use for OpenMP
		omp_set_num_threads(numThreads);

		// Compute summation in parallel
		#pragma omp parallel for
		for (int tId = 0; tId < numThreads; tId++)
		{
			// Initialize array for storing intermediate results for this particular thread
			results[tId] = new float[n];

			// Initialize results to 0
			memset(results[tId], 0.0, n*sizeof(float));

			// Determine which portion of the summation this thread will compute
			const int begin = tId * chunkSize;
			const int end = min(m, (tId+1)*chunkSize);

			// Compute summation for this thread's portion
			for (int i = begin; i < end; i++)
			{
				float mu = innerProduct(u[i], v[m], n);

				// Cache newly computed inner products
				if (i == m - 1)
				{
					cache[i] = innerProduct(u[i], u[i], n);
				}

				mu /= cache[i];

				// Store intermediate results (i.e. partial summation)
				for (int j = 0; j < n; j++)
				{
					results[tId][j] += (mu * u[i][j]);
				}
			}
		}

		// Aggregate partial summation results from each thread and finalize vector 'u'
		for (int tId = 0; tId < numThreads; tId++)
		{
			for (int j = 0; j < n; j++)
			{
				u[m][j] -= (results[tId][j]);
			}

			delete[] results[tId];
		}

		delete[] results;
	}

	// Perform normalization
	cache[n-1] = innerProduct(u[n-1], u[n-1], n);
	for (int m = 0; m < n; m++)
	{
		float norm = sqrt(cache[m]);
		for (int j = 0; j < n; j++)
		{
			u[m][j] /= norm;
		}
	}

	delete[] cache;
}

// Prints out the results of given matrix
void print(float** output, const int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << output[i][j];
			cout << " ";
		}
		cout << endl;
	}
}

int main(int argc, char *argv[])
{
	// Set seed for random number generation
	srand(1);

	// Get number of dimenions from command line
	const int numDim = atoi(argv[1]);

	// Get maximum number of threads to use from command line
	const int maxThreads = atoi(argv[2]);

	// Initialize matrices (collection of row vectors) 'v' and 'u'
	float** v = new float*[numDim];
	float** u = new float*[numDim];

	for (int i = 0; i < numDim; i++)
	{
		v[i] = new float[numDim];
		u[i] = new float[numDim];
	}

	// Fill input matrix 'v' with random numbers
	for (int i = 0; i < numDim; i++)
	{
		for (int j = 0; j < numDim; j++)
		{
			v[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}

	// Record start time
	auto startTime = chrono::high_resolution_clock::now();

	// Begin Gram-Schmidt process
	applyGramSchmidt(v, u, numDim, maxThreads);

	// Measure total elapsed time (in seconds)
	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	long duration = chrono::duration_cast<chrono::microseconds>(time).count();
	float sec = (float) duration / 1000000.0;

	// Output time
	cout << sec << endl;

	// Print output matrix
	// print(u, numDim);

	// Deallocate memory
	for (int i = 0; i < numDim; i++)
	{
		delete[] u[i];
		delete[] v[i];
	}

	delete[] u;
	delete[] v;

}
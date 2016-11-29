#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <omp.h>

using namespace std;

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

void applyGramSchmidt(float** v, float** u, const int n, const int threads) 
{
	float* cache = new float[n];

	copy(v[0], v[0]+n, u[0]);

	for (int m = 1; m < n; m++)
	{
		copy(v[m], v[m]+n, u[m]);

		const int numThreads = m < threads ? m : threads;

		const int chunkSize = (int)ceil((float)m/(float)numThreads);

		float** results = new float*[numThreads];

		omp_set_num_threads(numThreads);

		#pragma omp parallel for
		for (int tId = 0; tId < numThreads; tId++)
		{
			results[tId] = new float[n];

			memset(results[tId], 0.0, n*sizeof(float));

			const int begin = tId * chunkSize;
			const int end = min(m, (tId+1)*chunkSize);

			for (int i = begin; i < end; i++)
			{
				float mu = innerProduct(u[i], v[m], n);

				if (i == m - 1)
				{
					cache[i] = innerProduct(u[i], u[i], n);
				}

				mu /= cache[i];

				for (int j = 0; j < n; j++)
				{
					results[tId][j] += (mu * u[i][j]);
				}
			}
		}

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
	srand(1);

	const int numDim = atoi(argv[1]);
	const int threads = atoi(argv[2]);

	float** v = new float*[numDim];
	float** u = new float*[numDim];

	for (int i = 0; i < numDim; i++)
	{
		v[i] = new float[numDim];
		u[i] = new float[numDim];
	}

	for (int i = 0; i < numDim; i++)
	{
		for (int j = 0; j < numDim; j++)
		{
			v[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}

	auto startTime = chrono::high_resolution_clock::now();

	applyGramSchmidt(v, u, numDim, threads);

	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	long duration = chrono::duration_cast<chrono::microseconds>(time).count();
	float sec = (float) duration / 1000000.0;

	cout << sec << endl;

	// print(u, numDim);

	for (int i = 0; i < numDim; i++)
	{
		delete[] u[i];
		delete[] v[i];
	}

	delete[] u;
	delete[] v;

}
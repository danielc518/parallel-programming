#include <iostream>
#include <cmath>
#include <chrono>

/* Serial version of C++ code for comparison with OpenMP parallelization */

using namespace std;

// Computes inner product between vectors 'a' and 'b'
float innerProduct(float* a, float* b, const int n)
{
	float sum = 0.0;

	for (int i = 0; i < n; i++) 
	{
		sum += (a[i] * b[i]);
	}

	return sum;
}

// Applies Gram-Schmidt process to input matrix (row vectors) 'v' and outputs results to 'u'
void applyGramSchmidt(float** v, float** u, const int n) 
{
	// Set vector u_0 to v_0
	copy(v[0], v[0]+n, u[0]);

	// Begin iterations
	for (int m = 1; m < n; m++)
	{
		// Set vector u_m to v_m
		copy(v[m], v[m]+n, u[m]);

		// Subtract the summation terms
		for (int i = 0; i <= m - 1; i++)
		{
			float mu = innerProduct(u[i], v[m], n) / innerProduct(u[i], u[i], n);

			// Iterating over each component of a single vector
			for (int j = 0; j < n; j++)
			{
				u[m][j] -= (mu * u[i][j]);
			}
		}
	}

	// Perform normalization
	for (int m = 0; m < n; m++)
	{
		float norm = sqrt(innerProduct(u[m], u[m], n));
		for (int j = 0; j < n; j++)
		{
			u[m][j] /= norm;
		}
	}
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
	applyGramSchmidt(v, u, numDim);

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
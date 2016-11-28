#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

float innerProduct(float* a, float* b, const int n)
{
	float sum = 0.0;

	for (int i = 0; i < n; i++) 
	{
		sum += (a[i] * b[i]);
	}

	return sum;
}

void applyGramSchmidt(float** v, float** u, const int n) 
{
	copy(v[0], v[0]+n, u[0]);

	for (int m = 1; m < n; m++)
	{
		copy(v[m], v[m]+n, u[m]);

		for (int i = 0; i <= m - 1; i++)
		{
			float mu = innerProduct(u[i], v[m], n) / innerProduct(u[i], u[i], n);

			for (int j = 0; j < n; j++)
			{
				u[m][j] -= (mu * u[i][j]);
			}
		}
	}

	for (int m = 0; m < n; m++)
	{
		float norm = sqrt(innerProduct(u[m], u[m], n));
		for (int j = 0; j < n; j++)
		{
			u[m][j] /= norm;
		}
	}
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

	applyGramSchmidt(v, u, numDim);

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
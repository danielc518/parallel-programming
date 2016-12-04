import java.util.Arrays;
import java.util.Random;

public class Parallel {

	public static void main(String[] args) {

		// Initialize random number generator
		final Random random = new Random(31);

		// Get number of dimenions from command line
		final int numDim = Integer.parseInt(args[0]);

		// Get maximum number of threads to use from command line
		final int maxThreads = Integer.parseInt(args[1]);

		// Initialize matrices (collection of row vectors) 'v' and 'u'
		final double[][] v = new double[numDim][numDim];
		final double[][] u = new double[numDim][numDim];

		// Fill input matrix 'v' with random numbers
		for (int i = 0; i < numDim; i++) {
			for (int j = 0; j < numDim; j++) {
				v[i][j] = random.nextDouble();
			}
		}

		// Record start time
		long startTime = System.nanoTime();

		// Begin Gram-Schmidt process
		applyGramSchmidt(v, u, numDim, maxThreads);

		// Measure total elapsed time (in seconds)
		long elapsed = System.nanoTime() - startTime;

		double seconds = (double) elapsed / 1000000000.0;

		// Output time
		System.out.println(seconds);

		// print(u, numDim);
	}

	/**
	 * Applies Gram-Schmidt process to input data
	 * @param v Input data
	 * @param u Output data (i.e. orthogonalized data)
	 * @param n Number of dimensions
	 * @param maxThreads Maximum number of threads to be used
	 */
	public static void applyGramSchmidt(final double[][] v, final double[][] u, final int n, final int maxThreads) {
		final double[] cache = new double[n];

		u[0] = v[0];

		// Perform parallelized Gram-Schmidt process
		for (int m = 1; m < n; m++) {
			u[m] = v[m];

			// Check how many threads to use
			final int numThreads = m < maxThreads ? m : maxThreads;

			// Create array of threads
			final Thread[] threads = new Thread[numThreads];

			// Array to hold intermediate results
			final double[][] results = new double[numThreads][n];

			// Determine the size of work loads for each thread
			final int chunkSize = (int) Math.ceil(((double) m / (double) numThreads));
			
			// Thread ID
			int tId = 0;
			
			// Compute chunks of work loads in each thread
			for (int i = 0; i < numThreads; i++) {
				threads[tId] = new ComputeThread(tId, chunkSize, u, v[m], results[tId], n, m, cache);
				threads[tId].start();
				
				tId++;
			}

			// Synchronize the threads to aggregate intermediate results from each thread
			for (int i = 0; i < numThreads; i++) {
				try {
					threads[i].join();

					for (int j = 0; j < n; j++) {
						u[m][j] -= results[i][j];
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}

		// Normalize into unit vectors
		cache[n - 1] = innerProduct(u[n - 1], u[n - 1], n);
		for (int m = 0; m < n; m++) {
			double norm = Math.sqrt(cache[m]);
			for (int j = 0; j < n; j++) {
				u[m][j] /= norm;
			}
		}
	}

	/**
	 * Thread for computing a portion of summations required to compute orthogonalized vector
	 * @author Sanghyun
	 *
	 */
	public static class ComputeThread extends Thread {

		private final int id;
		private final int chunk;
		private final double[][] u;
		private final double[] v;
		private final double[] results;
		private final int n;
		private final int m;
		private final double[] cache;

		/**
		 * Main constructor for this thread
		 * @param id Thread ID
		 * @param chunk Work load chunk size
		 * @param u Current orthogonalized vectors
		 * @param v Input data
		 * @param results Array for holding intermediate results
		 * @param n Number of dimensions
		 * @param m Current iteration step
		 * @param cache Array for caching inner product results
		 */
		public ComputeThread(int id, int chunk, double[][] u, double[] v, 
				double[] results, int n, int m, double[] cache) {
			this.id = id;
			this.chunk = chunk;
			this.u = u;
			this.v = v;
			this.results = results;
			this.n = n;
			this.m = m;
			this.cache = cache;
		}

		@Override
		public void run() {
			Arrays.fill(results, 0.0);
			
			// Determine which portion of the summation this thread will compute
			int begin = id * chunk;
			int end = Math.min(m, (id + 1) * chunk);
			
			// Compute summation for this thread's portion
			for (int i = begin; i < end; i++) {
				double mu = innerProduct(u[i], v, n);

				// Cache newly computed inner products
				if (i == m - 1) {
					cache[i] = innerProduct(u[i], u[i], n);
				}

				mu /= cache[i];

				// Store intermediate results (i.e. partial summation)
				for (int j = 0; j < n; j++) {
					results[j] += (mu * u[i][j]);
				}
			}
		}
	}

	public static double innerProduct(final double[] a, final double[] b, final int n) {
		double sum1 = 0.0;
		double sum2 = 0.0;
		double sum3 = 0.0;
		double sum4 = 0.0;
		double sum5 = 0.0;

		for (int i = 0; i < n; i+=5) {
			sum1 += (a[i] * b[i]);
			sum2 += (a[i+1] * b[i+1]);
			sum3 += (a[i+2] * b[i+2]);
			sum4 += (a[i+3] * b[i+3]);
			sum5 += (a[i+4] * b[i+4]);
		}

		return sum1 + sum2 + sum3 + sum4 + sum5;
	}

	public static void print(final double[][] output, final int n) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				System.out.print(output[i][j] + " ");
			}
			System.out.println();
		}
	}

}

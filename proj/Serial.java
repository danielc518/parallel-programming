import java.util.Random;

public class Serial {

	public static void main(String[] args) {

		// Create random number generator for input data
		final Random random = new Random(31);

		// Get number of dimensions for each vector
		final int numDim = Integer.parseInt(args[0]);

		final double[][] v = new double[numDim][numDim]; // Input data
		final double[][] u = new double[numDim][numDim]; // Output data (i.e. orthogonalized)

		// Create artificial input data
		for (int i = 0; i < numDim; i++) {
			for (int j = 0; j < numDim; j++) {
				v[i][j] = random.nextDouble();
			}
		}

		long startTime = System.nanoTime();

		applyGramSchmidt(v, u, numDim);

		long elapsed = System.nanoTime() - startTime;

		double seconds = (double) elapsed / 1000000000.0;

		System.out.println(seconds);

		// print(u, numDim);
	}

	/**
	 * Applies Gram-Schmidt process to input data
	 * @param v Input data
	 * @param u Output data (i.e. orthogonalized data)
	 * @param n Number of dimensions
	 */
	public static void applyGramSchmidt(final double[][] v, final double[][] u, final int n) {
		u[0] = v[0];

		// Perform iterative Gram-Schmidt process
		for (int m = 1; m < n; m++) {
			u[m] = v[m];

			for (int i = 0; i <= m - 1; i++) {
				double mu = innerProduct(u[i], v[m], n) / innerProduct(u[i], u[i], n);

				for (int j = 0; j < n; j++) {
					u[m][j] -= (mu * u[i][j]);
				}
			}
		}

		// Normalize into unit vectors
		for (int m = 0; m < n; m++) {
			double norm = Math.sqrt(innerProduct(u[m], u[m], n));
			for (int j = 0; j < n; j++) {
				u[m][j] /= norm;
			}
		}
	}

	public static double innerProduct(final double[] a, final double[] b, final int n) {
		double sum = 0.0;

		for (int i = 0; i < n; i++) {
			sum += (a[i] * b[i]);
		}

		return sum;
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

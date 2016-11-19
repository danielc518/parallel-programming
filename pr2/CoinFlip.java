import java.util.Random;

public class CoinFlip {

	public static void main(String[] args) {
		// Check validity of arguments

		if (args == null || args.length != 2) {
			System.out.println("Usage: CoinFlip #threads #iterations");
			return;
		}

		if (!args[0].matches("[0-9]+") || !args[1].matches("[0-9]+")) {
			System.out.println("Only accepts integers as arguments!");
			return;
		}

		final int numThreads = Integer.parseInt(args[0]);
		final long numFlips = Long.parseLong(args[1]);

		runCoinFlip(numThreads, numFlips);
	}

	public static long runCoinFlip(final int numThreads, final long numFlips) {
		// Initialize parameters

		final CoinFlip coinFlip = new CoinFlip();

		final long flipsPerThread = (long) numFlips / numThreads;

		final CoinFlipThread[] threads = new CoinFlipThread[numThreads];

		long startTime = System.currentTimeMillis();

		// Create and start thread

		for (int i = 0; i < numThreads; i++) {
			threads[i] = coinFlip.new CoinFlipThread(flipsPerThread);
			threads[i].start();
		}

		long numHeads = 0;

		// Synchronize threads and get total number of heads

		for (int i = 0; i < numThreads; i++) {
			try {
				threads[i].join();
				numHeads += threads[i].numHeads;
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}

		long endTime = System.currentTimeMillis();
		long elapsedTime = endTime - startTime;

		// Output results

		System.out.format("%d heads in %d coin tosses.\nElapsed time: %dms\n", numHeads, numFlips, elapsedTime);

		return elapsedTime;
	}

	private class CoinFlipThread extends Thread {

		long numFlips = 0;
		long numHeads = 0;

		Random random = null;

		CoinFlipThread(long numFlips) {
			this.numFlips = numFlips;
			this.random = new Random();
		}

		@Override
		public void run() {
			if (this.numFlips == 0) {
				return;
			}

			for (int numFlip = 0; numFlip < numFlips; numFlip++) {
				this.numHeads += this.random.nextInt(2);
			}
		}
	}
}

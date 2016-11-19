import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Experiments {

	public static void main(String[] args) {
		PrintWriter pw = null;

		// 1. [Coin Flip] Speed-Up
		try {
			pw = new PrintWriter(new File("coinflip_speedup.csv"));

			final long numFlips = 1000000000;

			StringBuffer sb = new StringBuffer();

			for (int numThreads = 1; numThreads <= 32; numThreads *= 2) {
				sb.append(CoinFlip.runCoinFlip(numThreads, numFlips));
				sb.append(",");
			}

			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}

		// 2. [Coin Flip] Scale-Up
		try {
			pw = new PrintWriter(new File("coinflip_scaleup.csv"));

			final long numFlips = 1000000000;

			StringBuffer sb = new StringBuffer();

			for (int numThreads = 1; numThreads <= 32; numThreads *= 2) {
				sb.append(CoinFlip.runCoinFlip(numThreads, numFlips * numThreads));
				sb.append(",");
			}

			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}

		// 3. [Coin Flip] Start-Up
		try {
			pw = new PrintWriter(new File("coinflip_startup.csv"));

			StringBuffer sb = new StringBuffer();

			for (int numThreads = 100; numThreads <= 1000; numThreads += 100) {
				sb.append(CoinFlip.runCoinFlip(numThreads, 0));
				sb.append(",");
			}

			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}

		// 4. [Brute Force DES] Speed-Up
		try {
			pw = new PrintWriter(new File("des_speedup.csv"));

			final int keybits = 20;

			StringBuffer sb = new StringBuffer();

			for (int numThreads = 1; numThreads <= 32; numThreads *= 2) {
				sb.append(BruteForceDES.runBruteForceDES(numThreads, keybits));
				sb.append(",");
			}

			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}

		// 5. [Brute Force DES Scale-Up
		try {
			pw = new PrintWriter(new File("des_scaleup.csv"));

			int keybits = 20;

			StringBuffer sb = new StringBuffer();

			for (int numThreads = 1; numThreads <= 32; numThreads *= 2) {
				sb.append(BruteForceDES.runBruteForceDES(numThreads, keybits++));
				sb.append(",");
			}

			pw.write(sb.toString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}
	}
}

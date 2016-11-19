import java.util.Random;

import javax.crypto.Cipher;
import javax.crypto.SealedObject;
import javax.crypto.spec.SecretKeySpec;

public class BruteForceDES {

	public static void main(String[] args) {
		// Check validity of arguments

		if (args == null || args.length != 2) {
			System.out.println("Usage: BruteForceDES #threads key_size_in_bits");
			return;
		}

		if (!args[0].matches("[0-9]+") || !args[1].matches("[0-9]+")) {
			System.out.println("Usage: BruteForceDES #threads key_size_in_bits");
			return;
		}

		int numThreads = Integer.parseInt(args[0]);
		int keybits = Integer.parseInt(args[1]);

		runBruteForceDES(numThreads, keybits);
	}

	public static long runBruteForceDES(final int numThreads, final int keybits) {
		// Initialize parameters
		final BruteForceDES bruteForceDES = new BruteForceDES();

		long maxkey = ~(0L);
		maxkey = maxkey >>> (64 - keybits);

		// Create a simple cipher
		final SealedDES enccipher = bruteForceDES.new SealedDES();

		// Get a number between 0 and 2^64 - 1
		Random generator = new Random();
		long key = generator.nextLong();

		// Mask off the high bits so we get a short key
		key = key & maxkey;

		// Set up a key
		enccipher.setKey(key);

		System.out.format("Generated secret key %d\n", key);

		// Generate a sample string
		final String plainstr = "Johns Hopkins afraid of the big bad wolf?";

		SealedObject[] sldObjs = new SealedObject[numThreads];

		for (int i = 0; i < numThreads; i++) {
			sldObjs[i] = enccipher.encrypt(plainstr);
		}

		final long startTime = System.currentTimeMillis();
		final BruteForceThread[] threads = new BruteForceThread[numThreads];

		long keyPerThread = maxkey / numThreads;

		// Create and start threads

		for (int i = 0; i < numThreads; i++) {
			threads[i] = bruteForceDES.new BruteForceThread(i, i * keyPerThread, (i + 1) * keyPerThread, sldObjs[i]);
			threads[i].start();
		}

		// Synchronize threads

		for (int i = 0; i < numThreads; i++) {
			try {
				threads[i].join();
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}

		final long endTime = System.currentTimeMillis();
		long elapsedTime = endTime - startTime;
		
		// Output results
		
		System.out.format("Final elapsed time: %d\n", elapsedTime);

		return elapsedTime;
	}

	private class BruteForceThread extends Thread {

		long threadId;
		long startKey;
		long endKey;
		SealedObject sealedObj;

		BruteForceThread(long threadId, long startKey, long endKey, SealedObject sealedObj) {
			this.threadId = threadId;
			this.startKey = startKey;
			this.endKey = endKey;
			this.sealedObj = sealedObj;
		}

		@Override
		public void run() {
			final long startTime = System.currentTimeMillis();

			final SealedDES sealedDES = new SealedDES();

			for (long key = startKey; key < endKey; key++) {
				sealedDES.setKey(key);
				final String message = sealedDES.decrypt(sealedObj);

				if (message != null && message.contains("Hopkins")) {
					System.out.format("Thread %d Found decrypt key %d producing message: %s\n", threadId, key, message);
				}

				if (key % 100000 == 0) {
					final long endTime = System.currentTimeMillis();
					System.out.format("Thread %d Searched key number %d at %d milliseconds.\n", threadId, key,
							endTime - startTime);
				}
			}
		}
	}

	/**
	 * 
	 * Referenced from Professor Burns' "SealedDES.java" code
	 *
	 */
	class SealedDES {
		// Cipher for the class
		Cipher des_cipher;

		// Key for the class
		SecretKeySpec the_key = null;

		// Byte arrays that hold key block
		byte[] deskeyIN = new byte[8];
		byte[] deskeyOUT = new byte[8];

		// Constructor: initialize the cipher
		public SealedDES() {
			try {
				des_cipher = Cipher.getInstance("DES");
			} catch (Exception e) {
				System.out.println(
						"Failed to create cipher.  Exception: " + e.toString() + " Message: " + e.getMessage());
			}
		}

		// Decrypt the SealedObject
		//
		// arguments: SealedObject that holds on encrypted String
		// returns: plaintext String or null if a decryption error
		// This function will often return null when using an incorrect key.
		//
		public String decrypt(SealedObject cipherObj) {
			try {
				return (String) cipherObj.getObject(the_key);
			} catch (Exception e) {
				// System.out.println("Failed to decrypt message. " + ".
				// Exception: " + e.toString() + ". Message: " + e.getMessage())
				// ;
			}
			return null;
		}

		// Encrypt the message
		//
		// arguments: a String to be encrypted
		// returns: a SealedObject containing the encrypted string
		//
		public SealedObject encrypt(String plainstr) {
			try {
				des_cipher.init(Cipher.ENCRYPT_MODE, the_key);
				return new SealedObject(plainstr, des_cipher);
			} catch (Exception e) {
				System.out.println("Failed to encrypt message. " + plainstr + ". Exception: " + e.toString()
						+ ". Message: " + e.getMessage());
			}
			return null;
		}

		// Build a DES formatted key
		//
		// Convert an array of 7 bytes into an array of 8 bytes.
		//
		private void makeDESKey(byte[] in, byte[] out) {
			out[0] = (byte) ((in[0] >> 1) & 0xff);
			out[1] = (byte) ((((in[0] & 0x01) << 6) | (((in[1] & 0xff) >> 2) & 0xff)) & 0xff);
			out[2] = (byte) ((((in[1] & 0x03) << 5) | (((in[2] & 0xff) >> 3) & 0xff)) & 0xff);
			out[3] = (byte) ((((in[2] & 0x07) << 4) | (((in[3] & 0xff) >> 4) & 0xff)) & 0xff);
			out[4] = (byte) ((((in[3] & 0x0F) << 3) | (((in[4] & 0xff) >> 5) & 0xff)) & 0xff);
			out[5] = (byte) ((((in[4] & 0x1F) << 2) | (((in[5] & 0xff) >> 6) & 0xff)) & 0xff);
			out[6] = (byte) ((((in[5] & 0x3F) << 1) | (((in[6] & 0xff) >> 7) & 0xff)) & 0xff);
			out[7] = (byte) (in[6] & 0x7F);

			for (int i = 0; i < 8; i++) {
				out[i] = (byte) (out[i] << 1);
			}
		}

		// Set the key (convert from a long integer)
		public void setKey(long theKey) {
			try {
				// convert the integer to the 8 bytes required of keys
				deskeyIN[0] = (byte) (theKey & 0xFF);
				deskeyIN[1] = (byte) ((theKey >> 8) & 0xFF);
				deskeyIN[2] = (byte) ((theKey >> 16) & 0xFF);
				deskeyIN[3] = (byte) ((theKey >> 24) & 0xFF);
				deskeyIN[4] = (byte) ((theKey >> 32) & 0xFF);
				deskeyIN[5] = (byte) ((theKey >> 40) & 0xFF);
				deskeyIN[6] = (byte) ((theKey >> 48) & 0xFF);

				// theKey should never be larger than 56-bits, so this should
				// always be 0
				deskeyIN[7] = (byte) ((theKey >> 56) & 0xFF);

				// turn the 56-bits into a proper 64-bit DES key
				makeDESKey(deskeyIN, deskeyOUT);

				// Create the specific key for DES
				the_key = new SecretKeySpec(deskeyOUT, "DES");
			} catch (Exception e) {
				System.out.println("Failed to assign key" + theKey + ". Exception: " + e.toString() + ". Message: "
						+ e.getMessage());
			}
		}
	}
}

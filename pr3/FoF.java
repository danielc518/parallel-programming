import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class FoF {

	public static class Map extends Mapper<Object, Text, EdgeWritable, EdgeWritable> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer tokenizer = new StringTokenizer(value.toString());

			int user = Integer.parseInt(tokenizer.nextToken());

			List<Integer> friends = new ArrayList<Integer>();

			while (tokenizer.hasMoreTokens()) {
				friends.add(Integer.parseInt(tokenizer.nextToken()));
			}

			for (Integer friend : friends) {
				EdgeWritable edge = user < friend ? new EdgeWritable(user, friend) : new EdgeWritable(friend, user);
				for (Integer other : friends) {
					if ((int) friend != (int) other) {
						context.write(edge, new EdgeWritable(user, other));
					}
				}
			}
		}
	}

	public static class Reduce extends Reducer<EdgeWritable, EdgeWritable, NullWritable, Text> {
		public void reduce(EdgeWritable key, Iterable<EdgeWritable> values, Context context) throws IOException, InterruptedException {
			int userA = key.getLeft();
			int userB = key.getRight();

			Set<Integer> friendsOfA = new HashSet<Integer>();
			Set<Integer> friendsOfB = new HashSet<Integer>();

			for (EdgeWritable edge : values) {
				if (edge.getLeft() == userA) {
					friendsOfA.add(edge.getRight());
				} else if (edge.getLeft() == userB) {
					friendsOfB.add(edge.getRight());
				} else {
					continue;
				}
			}

			friendsOfA.retainAll(friendsOfB); // Find common friends

			for (Integer commonFriend : friendsOfA) {
				context.write(NullWritable.get(), new Text(commonFriend + " " + userA + " " + userB));
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Job job = Job.getInstance(new Configuration());
		job.setJarByClass(FoF.class);

		job.setJobName("FoF");

		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);

		job.setMapperClass(FoF.Map.class);
		job.setReducerClass(FoF.Reduce.class);

		job.setMapOutputKeyClass(EdgeWritable.class);
		job.setMapOutputValueClass(EdgeWritable.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.waitForCompletion(true);
	}

	public static class EdgeWritable implements WritableComparable<EdgeWritable> {

		private int left;
		private int right;

		public EdgeWritable() {
			set(left, right);
		}

		public EdgeWritable(int left, int right) {
			set(left, right);
		}

		public void set(int left, int right) {
			this.left = left;
			this.right = right;
		}

		public int getLeft() {
			return left;
		}

		public int getRight() {
			return right;
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(left);
			out.writeInt(right);
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			left = in.readInt();
			right = in.readInt();
		}

		@Override
		public int hashCode() {
			int a = left;
			int b = right;
			return a >= b ? a * a + a + b : a + b * b; // Szudzik's function
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj instanceof EdgeWritable) {
				EdgeWritable other = (EdgeWritable) obj;
				return other.getLeft() == this.left && other.getRight() == this.right;
			} else {
				return false;
			}
		}

		@Override
		public int compareTo(EdgeWritable obj) {
			int cmp = Integer.compare(this.left, obj.getLeft());

			if (cmp != 0) {
				return cmp;
			}

			return Integer.compare(this.right, obj.getRight());
		}

		@Override
		public String toString() {
			return left + "," + right;
		}
	}
}

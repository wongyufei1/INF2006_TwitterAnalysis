// Contributor: Derrick Lim

package topnegativereason;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;


public class ANSDriver {
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		Job job = Job.getInstance(conf, "Top 5 Negative Reasons in each Airline");
		job.setJarByClass(ANSDriver.class);
		Path inPath = new Path(otherArgs[0]);
		Path outPath = new Path(otherArgs[1]);
		outPath.getFileSystem(conf).delete(outPath, true);

		Configuration validationConf = new Configuration(false);
		ChainMapper.addMapper(job, ANSValidationMapper.class, LongWritable.class, Text.class, LongWritable.class, Text.class, validationConf);
		
		Configuration ansConf = new Configuration(false);
		ChainMapper.addMapper(job, ANSMapper.class, LongWritable.class, Text.class, Text.class, Text.class, ansConf);
		
		job.setMapperClass(ChainMapper.class);
		job.setReducerClass(ANSReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		

		FileInputFormat.addInputPath(job, inPath);
		FileOutputFormat.setOutputPath(job, outPath);
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}

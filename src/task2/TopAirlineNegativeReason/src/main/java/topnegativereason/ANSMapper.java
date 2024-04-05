package topnegativereason;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ANSMapper extends Mapper<LongWritable, Text, Text, Text> {

	private Text airline = new Text();
	private Text negativeReason = new Text();

	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String[] parts = value.toString().split(",");

		String reason = parts[15];
		String airlineName = parts[16];

		if (negativeReason != null && !negativeReason.equals("") && airlineName != null) {
			airline.set(airlineName);
			negativeReason.set(reason);

			context.write(airline, negativeReason);
		}
	}
}


// Contributor: Derrick Lim

package topnegativereason;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ANSMapper extends Mapper<LongWritable, Text, Text, Text> {

	private Text airline = new Text();
	private Text negativeReason = new Text();
    private int headerRowCount = 1;


	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		// Skip processing the header line
		if (key.get() < headerRowCount) {
            return;
        }
		String[] parts = value.toString().split(",");

		String reason = parts[7];
		String airlineName = parts[3];

		if (negativeReason != null && airlineName != null) {
			airline.set(airlineName);
			negativeReason.set(reason);

			context.write(airline, negativeReason);
		}
	}
}


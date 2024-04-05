package topnegativereason;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class ANSReducer extends Reducer<Text, Text, Text, Text> {
	private static final int TOP_K = 5;
	private Map<Text, TreeMap<Integer, String>> topReasonsMap = new HashMap<>();

	@Override
	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		TreeMap<Integer, String> topReasons = new TreeMap<>();

		// Count occurrences of each negative reason for this airline
		Map<String, Integer> reasonCounts = new HashMap<>();
		for (Text value : values) {
			String negReason = value.toString();
			if (negReason != null && !negReason.isEmpty()) {
				reasonCounts.put(negReason, reasonCounts.getOrDefault(negReason, 0) + 1);
			}
		}

		// Populate the TreeMap with the counts
		for (Map.Entry<String, Integer> entry : reasonCounts.entrySet()) {
			topReasons.put(entry.getValue(), entry.getKey());
		}

		// Keep only the top K negative reasons
		while (topReasons.size() > TOP_K) {
			topReasons.remove(topReasons.firstKey());
		}

		// Store the topReasons TreeMap for this airline
		topReasonsMap.put(new Text(key), topReasons);
	}

	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
		for (Map.Entry<Text, TreeMap<Integer, String>> entry : topReasonsMap.entrySet()) {
			Text airline = entry.getKey();
			TreeMap<Integer, String> topReasons = entry.getValue();

			// Emit the top K negative reasons for this airline
			int count = 0;
			for (Map.Entry<Integer, String> reasonEntry : topReasons.descendingMap().entrySet()) {
				if (count >= TOP_K) {
					break; // Stop emitting if we've emitted the top K reasons
				}
				context.write(airline, new Text(reasonEntry.getValue()));
				count++;
			}
		}
	}

}

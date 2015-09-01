package com.lcmf.rec.funds.indicator;

import java.io.IOException;
import java.io.PrintStream;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.lcmf.rec.funds.FundsCombination;
import com.lcmf.rec.io.db.FundValueReader;

public class FundProfit {

	public static double[] fundProfitRatioArray(double[] values) {

		int length = values.length;
		if (length <= 1)
			return null;
		double[] profits = new double[length - 1];
		for (int i = 0; i < length - 1; i++) {
			profits[i] = values[i + 1] / values[i] - 1;
		}
		return profits;
	}

	public static double fundProfitRatioAverage(double[] values) {

		int length = values.length;
		if (length <= 1)
			return 0.0;
		double sum = 0.0;
		for (int i = 0; i < length - 1; i++) {
			sum = sum + values[i + 1] / values[i] - 1;
		}
		double average_profit = sum / (length - 1);
		return average_profit;
	}

	public static void main(String[] args) throws IOException, SQLException, ParseException {
		// FundsCombination fc = new FundsCombination();
		// List<List<String>> fund_values =
		// fc.readCSVFundsData("./data/input/fund_values.csv");
		// PrintStream ps = new PrintStream("./data/tmp/profits.csv");
		// for(List<String> list : fund_values){
		// double[] values = FundsIndicator.removeEmptyValues(list);
		// double[] profits = fundProfitRatioArray(values);
		// StringBuilder sb = new StringBuilder();
		// for(double p : profits){
		// sb.append(p).append(",");
		// }
		// ps.println(sb.toString());
		// }
		// ps.close();

		FundValueReader fvReader = new FundValueReader();
		fvReader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		fvReader.readFundIds("./data/fund_pool/funds");
		fvReader.readFundValues("2006-01-04", "2015-05-30");

		PrintStream ps = new PrintStream("./data/tmp/profits.csv");
		HashMap<String, List<String>> map = fvReader.getFund_value_seq();
		for (String key : map.keySet()) {
			System.out.println(key);
			List<String> list = map.get(key);
			double[] values = FundsIndicator.removeEmptyValues(list);
			double[] profits = fundProfitRatioArray(values);
			StringBuilder sb = new StringBuilder();
			for (double p : profits) {
				sb.append(p).append(",");
			}
			ps.println(sb.toString());
		}
		ps.close();
	}
}

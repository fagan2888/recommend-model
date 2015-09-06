package com.lcmf.rec.funds.indicator;

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

}

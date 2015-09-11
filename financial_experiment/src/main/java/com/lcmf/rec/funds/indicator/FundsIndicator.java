package com.lcmf.rec.funds.indicator;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class FundsIndicator {
	
	private List<List<String>> fund_values   =  null;
	
	private double[][]         cleaned_data  =  null;
	
	private double[][]         cov           =  null;
	
	private double[]           returns       =  null;

	private double[]           variance      =  null;
	
	public FundsIndicator(List<List<String>> fund_values){
		this.fund_values = fund_values;
		this.cleaned_data = cleanData(fund_values);
		computeVariance();
		computeCov();
		computeReturns();
	}
	
	/**
	 * 去掉没有记录的日期的数据，取最小相交集
	 * @param list_values
	 * @return
	 */
	public static double[][] cleanData(List<List<String>> list_values){
		if(list_values == null ||  list_values.size() == 0){
			return null;
		}
		
		List<List<String>> result_list = new ArrayList<List<String>>();
		for(int i = 0; i < list_values.size(); i++){
			result_list.add(new ArrayList<String>());
		}
		
		int len = Integer.MAX_VALUE;
		for(int i = 0; i < list_values.size(); i++){
			if(list_values.get(i).size() < len){
				len = list_values.get(i).size();
			}
		}
		int num = list_values.size();
		
		for(int i = 0; i < len; i++){
			
			boolean all_has_value = true;
			for(int j = 0; j < num; j++){
				if("".equalsIgnoreCase(list_values.get(j).get(i)) || 0.0 == Double.parseDouble(list_values.get(j).get(i))){
					all_has_value = false;
					break;
				}
			}
			
			if(all_has_value){
				for(int j = 0; j < num; j++){
					result_list.get(j).add(list_values.get(j).get(i));
				}
			}
		}
		
		int r_len = result_list.get(0).size();
		int r_num = result_list.size();
		double[][] re = new double[r_num][r_len];
		for(int i = 0; i < result_list.size(); i++){
			for(int j = 0; j < result_list.get(i).size(); j++){
				re[i][j] = Double.parseDouble(result_list.get(i).get(j));
			}
		}
		return re;
	}
	

	/**
	 * compute every fund variance
	 * @throws FileNotFoundException 
	 */
	private void computeVariance(){
		variance = new double[cleaned_data.length];
		for(int i = 0; i < cleaned_data.length; i++){
			double[] values = cleaned_data[i];
			double[] profits = FundProfit.fundProfitRatioArray(values);
			if(profits == null){
				variance[i] = 0;
				continue;
			}else{
				variance[i] = COV.variance(profits);
			}
		}
	}

	/**
	 * compute every pair fund cov
	 * @throws FileNotFoundException 
	 */
	private void computeCov(){
		
		int len = cleaned_data.length;
		cov = new double[len][len];

		for(int i = 0; i < len; i++){
			for(int j = i; j < len; j++){
				cov[i][j] = COV.cov(FundProfit.fundProfitRatioArray(cleaned_data[i]), FundProfit.fundProfitRatioArray(cleaned_data[j]));
				cov[j][i] = cov[i][j];
			}
		}
		
	}
	
	private void computeReturns(){
	
		returns = new double[cleaned_data.length];
		for(int i = 0; i < cleaned_data.length; i++){
			double[] values = cleaned_data[i];
			if(values == null){
				returns[i] = 0;
				continue;
			}else{
				returns[i] = FundProfit.fundProfitRatioAverage(values);
			}
		}
	}

	
	/**
	 * compute funds values max retrance
	 * 
	 * @param values
	 * @return
	 */
	public static double maxRetrance(ArrayList<Double> values) {

		double max = 0.0; // max value
		double min = 0.0; // min value
		double retrance = 0.0; // retrance percent;

		for (double v : values) {

			if (v > max) {
				max = v;
				min = v;
			}

			if (v < min) {
				min = v;
			}

			double tmpRetrance = (max - min) / max;

			if (tmpRetrance > retrance) {
				retrance = tmpRetrance;
			}

		}

		return retrance;
	}
	
	public static double maxRetrance(double[] values) {

		double max = 0.0; // max value
		double min = 0.0; // min value
		double retrance = 0.0; // retrance percent;

		for (double v : values) {

			if (v > max) {
				max = v;
				min = v;
			}

			if (v < min) {
				min = v;
			}

			double tmpRetrance = (max - min) / max;

			if (tmpRetrance > retrance) {
				retrance = tmpRetrance;
			}

		}

		return retrance;
	}
	
	public double[][] getCov() {
		return cov;
	}
	
	public double[] getReturns() {
		return returns;
	}
	
	public double[] getVariance() {
		return variance;
	}
	
}

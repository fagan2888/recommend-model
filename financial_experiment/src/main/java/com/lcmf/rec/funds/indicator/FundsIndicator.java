package com.lcmf.rec.funds.indicator;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;


public class FundsIndicator {
	
	private List<List<String>> fund_values   =  null;
	
	private double[][]         cov           =  null;
	
	private double[]           returns       =  null;

	private double[]           variance      =  null;
	
	
	public FundsIndicator(List<List<String>> fund_values){
		this.fund_values = fund_values;
		computeVariance();
		computeCov();
		computeReturns();
	}
	
	/*
	 * remove empty value in string value list 
	 */
	public static double[] removeEmptyValues(List<String> str_values){

		List<Double> vs = new ArrayList<Double>();
		for(String s_value : str_values){
				if(s_value.equalsIgnoreCase("")){
					continue;
				}
				vs.add(Double.parseDouble(s_value));
		}
		if(vs.size() == 0){
			return null;
		}
		double[] ret = new double[vs.size()];
		for(int i = 0; i < vs.size(); i++){
			ret[i] = vs.get(i);
		}
		return ret;
	}

	public static double[][] removeEmptyValues(List<List<String>> list_values){
		if(list_values == null ||  list_values.size() == 0){
			return null;
		}
		
		List<List<String>> result_list = new ArrayList<List<String>>();
		for(int i = 0; i < list_values.size(); i++){
			result_list.add(new ArrayList<String>());
		}
		
		int len = list_values.get(0).size();
		int num = list_values.size();
		
		for(int i = 0; i < len; i++){
			
		}
		
		return null;
	} 
	

	/**
	 * compute every fund variance
	 * @throws FileNotFoundException 
	 */
	private void computeVariance(){
		variance = new double[fund_values.size()];
		for(int i = 0; i < fund_values.size(); i++){
			double[] values = removeEmptyValues(fund_values.get(i));
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
		computeVariance();
		
		int len = fund_values.size();
		cov = new double[len][len];
		
		for(int i = 0; i < len; i++){
			
			for(int j = i + 1; j < len; j++){
				DoubleFundValues dfv = removeEmptyValues(fund_values.get(i), fund_values.get(j));
				cov[i][j] = COV.cov(FundProfit.fundProfitRatioArray(dfv.fund_values1), FundProfit.fundProfitRatioArray(dfv.fund_values2));
				cov[j][i] = cov[i][j];
			}
		}
		
		for(int i = 0; i < len; i++){
			cov[i][i] = variance[i];
		}
	}
	
	private void computeReturns(){
	
		returns = new double[fund_values.size()];
		for(int i = 0; i < fund_values.size(); i++){
			double[] values = removeEmptyValues(fund_values.get(i));
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

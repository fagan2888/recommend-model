package com.lcmf.rec.funds.indicator;

/**
 * compute fund beta
 * @author yjiaoneal
 *
 */
public class FundBeta {

	
	public static final double beta(double[] fund_values, double[] benchmark_values){
		
		return COV.cov(fund_values, benchmark_values) / COV.cov(benchmark_values, benchmark_values);
	}
	
	public static void main(String[] args) {

	}

}

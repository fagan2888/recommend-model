package com.lcmf.rec.funds.indicator;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.lcmf.rec.io.db.FundValueReader;

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

	public static DoubleFundValues removeEmptyValues(List<String> str_values1, List<String> str_values2){
		if(str_values1 == null || str_values2 == null || str_values1.size() == 0 || str_values2.size() == 0 ){
			return null;
		}
		
		List<String> ret1 = new ArrayList<String>();
		List<String> ret2 = new ArrayList<String>();
		int len = str_values1.size() < str_values1.size()? str_values1.size(): str_values2.size();
		for(int i = 0; i < len; i++){
			String v1 = str_values1.get(i);
			String v2 = str_values2.get(i);
			if(v1.equalsIgnoreCase("") || v2.equalsIgnoreCase("")){
				continue;
			}
			ret1.add(v1);
			ret2.add(v2);
		}

		double[] d1 = new double[ret1.size()];
		double[] d2 = new double[ret2.size()];
		for(int i = 0; i < ret1.size(); i++){
			d1[i] = Double.parseDouble(ret1.get(i));
			d2[i] = Double.parseDouble(ret2.get(i));
		}
		
		DoubleFundValues dfv = new DoubleFundValues();
		dfv.fund_values1 = d1;
		dfv.fund_values2 = d2;
		return dfv;
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
	
	public double[][] getCov() {
		return cov;
	}
	
	public double[] getReturns() {
		return returns;
	}
	
	public double[] getVariance() {
		return variance;
	}
	
	public static void main(String[] args) throws SQLException, FileNotFoundException, ParseException {

		FundValueReader fvReader = new FundValueReader();
    	fvReader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		fvReader.readFundIds("./data/fund_pool/funds");
		fvReader.readFundValues("2006-01-04", "2015-05-30");
		
		List<List<String>> values = new ArrayList<List<String>>();
		HashMap<String, List<String>> map = fvReader.getFund_value_seq();
		for(String key: map.keySet()){
			System.out.println(key);
			values.add(map.get(key));
		}
		
		FundsIndicator fi = new FundsIndicator(values);
		fi.computeCov();
		fi.computeReturns();
		
		double[][] cov = fi.getCov();
		double[]  res  = fi.getReturns();
		PrintStream ps = new PrintStream("./data/tmp/cov.csv");
		for(int i = 0; i < cov.length; i++){
			StringBuilder sb = new StringBuilder();
			for(int j = 0; j < cov[i].length ;j++){
				sb.append(cov[i][j]).append(",");
			}
			ps.println(sb.toString());
		}
		ps.close();
		ps = new PrintStream("./data/tmp/res");
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < res.length; i++){
			sb.append(res[i]).append(",");
		}
		ps.println(sb.toString());
		ps.close();
	}

}

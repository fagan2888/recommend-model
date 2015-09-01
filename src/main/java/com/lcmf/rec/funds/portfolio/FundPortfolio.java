package com.lcmf.rec.funds.portfolio;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.funds.FundsCombination;
import com.lcmf.rec.funds.indicator.FundMaxYield;
import com.lcmf.rec.funds.markowitz.FrontierPoint;

public class FundPortfolio {

	private FundsCombination fc = null;  /** 计算该资产组合的类 */
	
	private FrontierPoint fp = null;     /** 有效前沿的点 */
	
	private String fp_name = "";	     /** 资产组合名字 */

	private String type = "etf";			 /** 资产组合类型 */
	
	private double[] fpValues = null;     /**  资产组合的净值走势*/
	
	private double maxDrawdonw = 0.0;    /**   最大回撤    */
	
	private double total_return_ratio = 0.0; /** 累计收益*/
	
	private double annual_return_ratio = 0.0; /** 年化收益*/
	
	private String riskvsreturn = "风险与收益相匹配";

	public FundPortfolio(FrontierPoint fp, String type, List<List<String>> vlist){
		this.type = type;
		this.fp = fp;
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String today_str = format.format(new Date());
		String fp_id_str = String.format("%s%03d", today_str, Math.round(this.fp.getRisk_grade() * 100));
		this.fp_name = type + "_" + fp_id_str;
		
		/** 计算组合的净值*/
		this.fpValues = computePerformanceValues(vlist);
		/** 计算组合的最大回撤*/
		this.computeDrawdown();
		/** 计算年化收益 */
		this.computeAnnualReturnRatio();
	}

	public double getCampSd(){
		return this.fp.getCamp_sd();
	}
	
	public double getCampReturn(){
		return this.fp.getCamp_return();
	}

	public double getMaxDrawdonw() {
		if(0.0 == this.maxDrawdonw){
			computeDrawdown();
		}
		return maxDrawdonw;
	}
	

	/**
	 * 预期5年最大值
	 * @return
	 */
	public float expectReturnMax(){
		
		double u = this.fp.getCamp_return();
		double sigma = this.fp.getCamp_sd();
		
		int days = 250 * 5;
		double p = Math.pow(Math.E, u * days);
		double upper_p = p * (1 + sigma * Math.sqrt(days)) - 1;
		return (float)upper_p;
	}
	
	
	/**
	 * 预期5年最小值
	 * @return
	 */
	public float expectReturnMin(){
		
		double u = this.fp.getCamp_return();
		double sigma = this.fp.getCamp_sd();
		
		int days = 250 * 5;
		double p = Math.pow(Math.E, u * days);
		double bottom_p = p * (1 - sigma * Math.sqrt(days)) - 1;
		return (float)bottom_p;
	}

	/**
	 * 计算最大回撤
	 */
	public void computeDrawdown() {
		this.maxDrawdonw = FundMaxYield.maxYield(this.fpValues);
	}
	
	/**
	 * 计算年化收益
	 * @return
	 */
	public double computeAnnualReturnRatio() {
		double head = fpValues[0];
		double tail = fpValues[fpValues.length - 1];
		int len = fpValues.length;
		this.annual_return_ratio = Math.pow(tail / head, 1 / (1.0 * len / 365)) - 1;
		return this.annual_return_ratio;
	}
	
	public double getRiskGrade(){
		return fp.getRisk_grade();
	}
	
	/**
	 * 计算累计收益
	 * @return
	 */
	public double computeTotalReturnRatio(){
		this.total_return_ratio = fpValues[fpValues.length - 1] / fpValues[0] - 1;
		return this.total_return_ratio;
	}
	
	public double[] computePerformanceValues(List<List<String>> pValues) {

		double[] fpValues = null;
		for (List<String> list : pValues) {
			fpValues = new double[list.size()];
		}
		fpValues[0] = 1.0;

		double[] weights = this.fp.getWeights();
		int len = fpValues.length;
		int num = pValues.size();

		for (int i = 1; i < len; i++) {
			double profit = 0.0;
			double sum_w = 0.0;
			for (int j = 0; j < num; j++) {
				double w = weights[j];
				sum_w += w;
				String today_value_str = pValues.get(j).get(i);
				int m = i - 1;
				if (today_value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(today_value_str)) {
					continue;
				}
				while (m >= 0) {
					String value_str = pValues.get(j).get(m);
					if (value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(value_str)) {
						m--;
					} else {
						profit += w * (Double.parseDouble(today_value_str) / Double.parseDouble(value_str) - 1);
						break;
					}
				}
			}
			profit = profit * sum_w + ConstVarManager.getRf() * (1 - sum_w);
			fpValues[i] = fpValues[i - 1] * (profit + 1);
		}
		
		return fpValues;
	}
	
	
	
	public double[] getFpValues() {
		return fpValues;
	}

	public double[] getWeights(){
		return this.fp.getWeights();
	}
	
	public double getTotal_return_ratio() {
		return total_return_ratio;
	}

	public double getAnnual_return_ratio() {
		return annual_return_ratio;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getFp_name() {
		return fp_name;
	}
	
	public void setFp_name(String fp_name) {
		this.fp_name = fp_name;
	}

	
	public String getRiskvsreturn() {
		return riskvsreturn;
	}

	public void setRiskvsreturn(String riskvsreturn) {
		this.riskvsreturn = riskvsreturn;
	}

	public static void main(String[] args) {

	}

}

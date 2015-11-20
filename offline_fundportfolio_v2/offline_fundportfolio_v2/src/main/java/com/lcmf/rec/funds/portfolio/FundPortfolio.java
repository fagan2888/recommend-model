package com.lcmf.rec.funds.portfolio;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.apache.log4j.Logger;
import com.lcmf.rec.funds.indicator.COV;
import com.lcmf.rec.funds.indicator.FundMaxYield;
import com.lcmf.rec.funds.markowitz.FrontierPoint;


public class FundPortfolio {

	private static Logger logger = Logger.getLogger(FundPortfolio.class);

	private FrontierPoint fp = null;
	/** 有效前沿的点 */

	private double[] fpValues = null;
	/** 资产组合的净值走势 */

	private double maxDrawdonw = 0.0;
	/** 最大回撤 */

	private double annual_return_ratio = 0.0;
	/** 年化收益 */

	private double expect_returns_max = 0.0;
	/** 预期年化收益最大值 */
	
	private double expect_returns_min = 0.0;
	/** 预期年化收益最小值 */
	
	private double u = 0.0;
	
	private double std = 0.0;
	
	private double risk = -1;
	
	private String riskvsreturn = "风险与收益相匹配";
	
	private String name = "";
	
	private String type = "etf";
	
	private String risk_name = "etf";
	
	private List<String> mofang_ids = new ArrayList<String>();
	
	private List<Double> weights = new ArrayList<Double>();
	
	private double total_return_ratio = 0;

	public FundPortfolio(String name, double[] fpValues) {
		
		this.name = name;
		
		this.fpValues = fpValues;

		/** 计算组合的最大回撤 */
		this.computeDrawdown();

		/** 计算收益和风险 */
		this.computeRiskReturn();
		
		total_return_ratio = fpValues[fpValues.length - 1] / fpValues[0] - 1;
		
	}

	
	private void computeRiskReturn(){
		
		double[] profits = new double[fpValues.length - 1];
		
		double profit_sum = 0.0;
		for(int i = 1; i < fpValues.length; i++){
			profit_sum = profit_sum + (fpValues[i] / fpValues[i-1] - 1);
			profits[i - 1] = fpValues[i] / fpValues[i-1] - 1;
		}
		u = profit_sum / (fpValues.length - 1);
		std = Math.sqrt(COV.variance(profits));
	}
	

	public void generateRiskName(){
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String today_str = format.format(new Date());
		String fp_id_str = String.format("%s%03d", today_str, Math.round(this.getRisk() * 100));
		this.risk_name = type + "_" + fp_id_str;
	}
	
	public double getMaxDrawdonw() {
		if (0.0 == this.maxDrawdonw) {
			computeDrawdown();
		}
		return maxDrawdonw;
	}

	/**
	 * 计算最大回撤
	 */
	public void computeDrawdown() {
		this.maxDrawdonw = FundMaxYield.maxYield(this.fpValues);
	}

	
	/**
	 * 计算年回报率
	 */
	public void computeAnnualReturns(){
		
		computeRiskReturn();
		int len = fpValues.length;
		annual_return_ratio = Math.pow(total_return_ratio + 1, 1.0 / (1.0 * len / 250)) - 1;
		double portfolio_p =  Math.pow(Math.E, this.u * 250);
		expect_returns_max = portfolio_p * ( 1 + this.std * Math.sqrt(250)) - 1;
		expect_returns_min = portfolio_p * ( 1 - this.std * Math.sqrt(250)) - 1;
		
	}

	public double[] getFpValues() {
		return fpValues;
	}

	public double[] getWeights() {
		double[] ws = new double[weights.size()];
		for(int i = 0; i < ws.length; i++){
			ws[i] = weights.get(i);
		}
		return ws;
	}

	public double getAnnual_return_ratio() {
		return annual_return_ratio;
	}

	public double getExpect_returns_max() {
		return expect_returns_max;
	}

	public void setExpect_returns_max(double expect_returns_max) {
		this.expect_returns_max = expect_returns_max;
	}

	public double getExpect_returns_min() {
		return expect_returns_min;
	}

	public void setExpect_returns_min(double expect_returns_min) {
		this.expect_returns_min = expect_returns_min;
	}

	public String getRiskvsreturn() {
		return riskvsreturn;
	}

	public void setRiskvsreturn(String riskvsreturn) {
		this.riskvsreturn = riskvsreturn;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void setMaxDrawdonw(double maxDrawdonw) {
		this.maxDrawdonw = maxDrawdonw;
	}

	public void setAnnual_return_ratio(double annual_return_ratio) {
		this.annual_return_ratio = annual_return_ratio;
	}

	public static void main(String[] args) {

	}

	public double getU() {
		return u;
	}

	public void setU(double u) {
		this.u = u;
	}

	public double getStd() {
		return std;
	}

	public void setStd(double std) {
		this.std = std;
	}

	public double getRisk() {
		return risk / 10;
	}

	public void setRisk(double risk) {
		this.risk = risk;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getRisk_name() {
		return risk_name;
	}

	public void setRisk_name(String risk_name) {
		this.risk_name = risk_name;
	}

	public List<String> getMofang_ids() {
		return mofang_ids;
	}

	public void setMofang_ids(List<String> mofang_ids) {
		this.mofang_ids = mofang_ids;
	}

	public void setFpValues(double[] fpValues) {
		this.fpValues = fpValues;
	}

	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}

	public double getTotal_return_ratio() {
		return total_return_ratio;
	}
	
}
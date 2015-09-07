package com.lcmf.rec.funds.portfolio;

public class FundPortfolio {


	/** 资产组合收益*/
	private double portfolio_return = 0.0;

	/** 资产组合风险 */
	private double protfolio_risk   = 0.0;
	
	/** 各个资产权重*/
	private double[] weights        = null;
	
	/** 资产组合名字 */
	private String fp_name = "";

	public double getPortfolio_return() {
		return portfolio_return;
	}

	public void setPortfolio_return(double portfolio_return) {
		this.portfolio_return = portfolio_return;
	}

	public double getProtfolio_risk() {
		return protfolio_risk;
	}

	public void setProtfolio_risk(double protfolio_risk) {
		this.protfolio_risk = protfolio_risk;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}

	public String getFp_name() {
		return fp_name;
	}

	public void setFp_name(String fp_name) {
		this.fp_name = fp_name;
	}

}
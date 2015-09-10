package com.lcmf.rec.funds.markowitz;

public class FrontierPoint {

	private double[] weights;
	private double   camp_return;
	private double   camp_risk;

	public FrontierPoint(double camp_return, double camp_sd, double[] weights){
		this.camp_return = camp_return;
		this.camp_risk = camp_sd;
		this.weights = weights;
	}
	
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for(double v : weights){
			sb.append(v).append(",");
		}
		sb.append(",");
		sb.append(camp_risk).append(",");
		sb.append(camp_return);
		return sb.toString();
	}
	
	public FrontierPoint clone(){
		double[] ws = new double[weights.length];
		for(int i = 0; i < weights.length; i++){
			ws[i] = weights[i];
		}
		FrontierPoint fp = new FrontierPoint(this.camp_return, this.camp_risk, this.weights);
		return fp;
	}


	public double[] getWeights() {
		return weights;
	}


	public void setWeights(double[] weights) {
		this.weights = weights;
	}


	public double getCamp_return() {
		return camp_return;
	}


	public void setCamp_return(double camp_return) {
		this.camp_return = camp_return;
	}


	public double getCamp_risk() {
		return camp_risk;
	}


	public void setCamp_risk(double camp_risk) {
		this.camp_risk = camp_risk;
	}

	
}

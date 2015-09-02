package com.lcmf.rec.funds.markowitz;

public class FrontierPoint {

	private double[] weights;
	private double   camp_return;
	private double   camp_sd;
	private double   risk_grade = -1;

	
	public FrontierPoint(double camp_return, double camp_sd, double[] weights, double risk_grade){
		this.camp_return = camp_return;
		this.camp_sd = camp_sd;
		this.weights = weights;
		this.risk_grade = risk_grade;
	}
	
	public FrontierPoint(double camp_return, double camp_sd, double[] weights){
		this.camp_return = camp_return;
		this.camp_sd = camp_sd;
		this.weights = weights;
	}
	
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for(double v : weights){
			sb.append(v).append(",");
		}
		sb.append(",");
		sb.append(camp_sd).append(",");
		sb.append(camp_return);
		return sb.toString();
	}
	
	public FrontierPoint clone(){
		double[] ws = new double[weights.length];
		for(int i = 0; i < weights.length; i++){
			ws[i] = weights[i];
		}
		FrontierPoint fp = new FrontierPoint(this.getCamp_return(), this.getCamp_sd(), ws, this.getRisk_grade());
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

	public double getCamp_sd() {
		return camp_sd;
	}

	public void setCamp_sd(double camp_sd) {
		this.camp_sd = camp_sd;
	}

	public double getRisk_grade() {
		return risk_grade;
	}

	public void setRisk_grade(double risk_grade) {
		this.risk_grade = risk_grade;
	}
	
}

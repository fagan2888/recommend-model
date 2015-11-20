package com.lcmf.rec.funds.markowitz;

import static org.ojalgo.constant.BigMath.*;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.ojalgo.optimisation.Expression;
import org.ojalgo.optimisation.ExpressionsBasedModel;
import org.ojalgo.optimisation.Optimisation;
import org.ojalgo.optimisation.Variable;

public class Markowitz {

	private double[] camp_return = null;

	private double[][] camp_covariance = null;

	private int num = 1000;
	
	
	
	public Markowitz() {
		
	}

	public Markowitz(double[] camp_return, double[][] camp_covariance) {
		this.camp_covariance = camp_covariance;
		this.camp_return = camp_return;
	}

	/**
	 * compute perfect shape point
	 * @param points
	 * @param rf
	 * @return
	 */
	public static FrontierPoint perfectShape(List<FrontierPoint> points, double rf) {
		
		double max_slope = -Double.MAX_VALUE;
		FrontierPoint max_point = null;
		
		for(FrontierPoint point : points){
			double tmp_slope = (point.getCamp_return() - rf) / point.getCamp_sd();
			if (tmp_slope > max_slope){
				max_slope = tmp_slope;
				max_point = point;
			}
		}
		return max_point;
	}
	
	
	public static FrontierPoint minPoint(List<FrontierPoint> points){
		
		double min_var = Double.MAX_VALUE;
		FrontierPoint min_point = null;
		for(FrontierPoint point : points){
			if(point.getCamp_sd() < min_var){
				min_point = point;
				min_var = point.getCamp_sd();
			}
		}
		
		return min_point;
	}
	

	
	/**
	 * compute efficient frontier
	 * 
	 * @param camp_return
	 * @param camp_covariance
	 * @param num
	 * @return
	 * @throws FileNotFoundException 
	 */
	public List<FrontierPoint> efficientFrontier() {

		final String asset_name = "Asset-";

		int camp_length = camp_return.length;

		List<FrontierPoint> results = new ArrayList<FrontierPoint>();

		double max_return = Double.MIN_VALUE;
		double min_return = Double.MAX_VALUE;

		for (int i = 0; i < camp_length; i++) {
			double v = camp_return[i];
			if (v > max_return)
				max_return = v;
			if (v < min_return)
				min_return = v;
		}

		double interval = (max_return - min_return) / num;
		double[] res = new double[num];
		for (int i = 0; i < num; i++) {
			res[i] = min_return + interval * i;
		}

		Variable[] tmpVariables = new Variable[camp_length];
		for (int i = 0; i < tmpVariables.length; i++) {
			tmpVariables[i] = (new Variable(asset_name + String.valueOf(i))).lower(ZERO);
		}

		ExpressionsBasedModel ebm = new ExpressionsBasedModel(tmpVariables);

		Expression weights_express = ebm.addExpression("Weights");
		for (int i = 0; i < camp_length; i++) {
			weights_express.setLinearFactor(i, ONE);
		}
		weights_express.level(ONE);

		Expression variable_express = ebm.addExpression("Variables");
		for (int j = 0; j < camp_length; j++) {
			for (int i = 0; i < camp_length; i++) {
				variable_express.setQuadraticFactor(j, i, camp_covariance[j][i]);
			}
		}
		variable_express.weight(ONE);

		for (int j = 0; j < num; j++) {

			Expression returns_express = ebm.addExpression("Returns");
			for (int i = 0; i < camp_length; i++) {
				returns_express.setLinearFactor(i, camp_return[i]);
			}
			returns_express.level(res[j]);

			
			Optimisation.Result tmpResult = ebm.minimise();
			if (tmpResult.getState().isOptimal() && tmpResult.getState().isSuccess()) {
				double camp_sd = Math.sqrt(tmpResult.getValue());
				double camp_return = res[j];
				double[] ws = new double[(int) tmpResult.count()];
				for (int n = 0; n < tmpResult.count(); n++) {
					ws[n] = tmpResult.doubleValue(n);
				}
				FrontierPoint markResult = new FrontierPoint(camp_return, camp_sd, ws);
				results.add(markResult);
			}
		}
		
		return results;
	}


	
	public static void main(String[] args) throws FileNotFoundException {

		final double[] camp_return = new double[] { 0.000202, 0.001804, 0.055754, 0.033945, 0.065950, 0.031631,
				0.039204, 0.056023 };

		final double[][] camp_covariance = new double[][] {
				{ 0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437 },
				{ 0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535 },
				{ -0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039 },
				{ -0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854 },
				{ 0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994 },
				{ 0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235 },
				{ -0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064 },
				{ -0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.0799 }, 
		};
				
		long start = System.currentTimeMillis();
		Markowitz mark = new Markowitz(camp_return, camp_covariance);
		List<FrontierPoint> results = mark.efficientFrontier();
		long end = System.currentTimeMillis();
		
		System.out.println(end - start);
	}

	public double[] getCamp_return() {
		return camp_return;
	}

	public void setCamp_return(double[] camp_return) {
		this.camp_return = camp_return;
	}

	public double[][] getCamp_covariance() {
		return camp_covariance;
	}

	public void setCamp_covariance(double[][] camp_covariance) {
		this.camp_covariance = camp_covariance;
	}

	public int getNum() {
		return num;
	}

	public void setNum(int num) {
		this.num = num;
	}
	

}

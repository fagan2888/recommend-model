package com.lcmf.rec.funds.markowitz;

import static org.ojalgo.constant.BigMath.ONE;
import static org.ojalgo.constant.BigMath.ZERO;
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

	public Markowitz(double[] camp_return, double[][] camp_covariance) {
		this.camp_covariance = camp_covariance;
		this.camp_return = camp_return;
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

	/** 固定收益率，计算风险最小的投资组合*/
	public FrontierPoint targetReturn(double t_return) {

		final String asset_name = "Asset-";
		FrontierPoint markResult = null;

		int camp_length = camp_return.length;

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

		Expression returns_express = ebm.addExpression("Returns");
		for (int i = 0; i < camp_length; i++) {
			returns_express.setLinearFactor(i, camp_return[i]);
		}
		returns_express.level(t_return);

		Optimisation.Result tmpResult = ebm.minimise();
		if (tmpResult.getState().isOptimal() && tmpResult.getState().isSuccess()) {
			double camp_sd = Math.sqrt(tmpResult.getValue());
			double camp_return = t_return;
			double[] ws = new double[(int) tmpResult.count()];
			for (int n = 0; n < tmpResult.count(); n++) {
				ws[n] = tmpResult.doubleValue(n);
			}
			markResult = new FrontierPoint(camp_return, camp_sd, ws);
		}
		return markResult;
	}

	
	/** 固定收益率，计算风险最小的投资组合*/
	public FrontierPoint pca_targetReturn(double t_return, double[][] eigen_vector) {

		final String asset_name = "Asset-";
		FrontierPoint markResult = null;

		int camp_length = camp_return.length;

		Variable[] tmpVariables = new Variable[camp_length];
		for (int i = 0; i < tmpVariables.length; i++) {
			tmpVariables[i] = (new Variable(asset_name + String.valueOf(i)));
		}

		ExpressionsBasedModel ebm = new ExpressionsBasedModel(tmpVariables);

		int len = eigen_vector.length;
		for(int i = 0; i < len; i++){
			Expression weights_lowwer_express = ebm.addExpression("Weights_lowwer_" + String.valueOf(i));
			double[] vec = eigen_vector[i];
			for(int j = 0; j < vec.length; j++){
				weights_lowwer_express.setLinearFactor(j, vec[j]);
			}
			weights_lowwer_express.lower(ZERO);
		}
		
		int num = eigen_vector[0].length;
		double[] wws = new double[num];
		for(int i = 0; i < num; i++){
			for(int j = 0; j < len; j++){
				wws[i] = wws[i] + eigen_vector[j][i];
			}
		}
		Expression weights_express = ebm.addExpression("Weights");
		for (int i = 0; i < num; i++) {
			weights_express.setLinearFactor(i, wws[i]);
		}
		weights_express.level(ONE);
		
		Expression variable_express = ebm.addExpression("Variables");
		for (int j = 0; j < camp_length; j++) {
			for (int i = 0; i < camp_length; i++) {
				variable_express.setQuadraticFactor(j, i, camp_covariance[j][i]);
			}
		}
		variable_express.weight(ONE);

		Expression returns_express = ebm.addExpression("Returns");
		for (int i = 0; i < camp_length; i++) {
			returns_express.setLinearFactor(i, camp_return[i]);
		}
		returns_express.level(t_return);

		Optimisation.Result tmpResult = ebm.minimise();
		if (tmpResult.getState().isOptimal() && tmpResult.getState().isSuccess()) {
			double camp_sd = Math.sqrt(tmpResult.getValue());
			double camp_return = t_return;
			double[] ws = new double[(int) tmpResult.count()];
			for (int n = 0; n < tmpResult.count(); n++) {
				ws[n] = tmpResult.doubleValue(n);
			}
			markResult = new FrontierPoint(camp_return, camp_sd, ws);
		}
		return markResult;
	}

	public void setNum(int num) {
		this.num = num;
	}
	
}
package com.lcmf.rec.funds.pca;

import static org.ojalgo.constant.BigMath.ONE;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.List;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Test;
import org.ojalgo.optimisation.Expression;
import org.ojalgo.optimisation.ExpressionsBasedModel;
import org.ojalgo.optimisation.Optimisation;
import org.ojalgo.optimisation.Variable;
import com.lcmf.rec.funds.indicator.COV;
import com.lcmf.rec.funds.indicator.FundProfit;
import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.io.OUT;
import com.lcmf.rec.funds.markowitz.FrontierPoint;
import com.lcmf.rec.funds.markowitz.Markowitz;
import com.lcmf.rec.io.db.FundValueReader;
import Jama.Matrix;

public class TestPCA {

	static {
		PropertyConfigurator.configure("./conf/log4j.properties");
	}

	public void testPCA() {
		final double[][] covariance = new double[][] {
				{ 0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437 },
				{ 0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535 },
				{ -0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039 },
				{ -0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854 },
				{ 0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994 },
				{ 0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235 },
				{ -0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064 },
				{ -0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.0799 }, };

		PCA pca = new PCA(covariance);
		pca.getPc_matrix();
	}
	
	//@Test
	public void testPCAAccuracy()
			throws SQLException, ParseException, FileNotFoundException, UnsupportedEncodingException {

		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();

		int num = ds.size();
		int len = ds.get(0).size();
		double[][] tmp_datas = FundsIndicator.cleanData(ds);
		double[][] profits = new double[num][len - 1];
		for (int i = 0; i < num; i++) {
			profits[i] = FundProfit.fundProfitRatioArray(tmp_datas[i]);
		}
		double[][] datas = new Matrix(profits).transpose().getArrayCopy();
//		OUT.printStdout(COV.cov(datas));

		System.out.println("-------------");
		PCA pca = new PCA(datas);
		Matrix cov = pca.getEigen_vector().times(pca.getEigen_diagonal()).times(pca.getEigen_vector().transpose());
		
		OUT.printStdout(cov.getArrayCopy());
		System.out.println("--------------");
		OUT.printStdout(pca.getPca_vector_matrix().times(pca.getPca_value_diagonal()).times(pca.getPca_vector_matrix().transpose()).getArrayCopy());
		System.out.println("---------------");
		OUT.printStdout(pca.getEigen_diagonal().getArrayCopy());
		System.out.println("---------------");
		
//		OUT.printStdout(cov.getArrayCopy());
//		OUT.printStdout(COV.cov(pca.getPc_matrix().transpose().getArrayCopy()));
//		System.out.println("-------------");
//		OUT.printStdout(pca.getEigen_diagonal().getArrayCopy());
//		OUT.printStdout(cov.getArrayCopy());
		
//		System.out.println("-------------");
//		Matrix pca_cov = pca.getPca_vector_matrix().times(pca.getPca_value_diagonal()).times(pca.getPca_vector_matrix().transpose());
//		OUT.printStdout(pca_cov.getArrayCopy());
//
//		System.out.println("-------------");
//		OUT.printStdout(pca.getEigen_vector().getArrayCopy());
//		System.out.println("-------------");
//		OUT.printStdout(pca.getEigen_vector().inverse().getArrayCopy());
//		System.out.println("--------------");
//		OUT.printStdout(pca.getPca_vector_matrix().transpose().times(pca.getPca_vector_matrix()).getArrayCopy());
//		OUT.printStdout(pca.getPca_vector_matrix().getArrayCopy());
//		System.out.println("--------------");
//		OUT.printStdout(pca.getEigen_diagonal().getArrayCopy());
//		System.out.println("---------------");
//		OUT.printStdout(pca.getPca_vector_matrix().times(pca.getPca_value_diagonal()).times(pca.getPca_vector_matrix().inverse()).getArrayCopy());
//		System.out.println("---------------");
//		OUT.printStdout(pca.getPca_vector_matrix().times(pca.getPca_vector_matrix().inverse()).getArrayCopy());
		
		
	}

	
	//@Test
	public void testETFPCA() throws SQLException, ParseException, FileNotFoundException, UnsupportedEncodingException {

		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();

		int num = ds.size();
		int len = ds.get(0).size();
		double[][] tmp_datas = FundsIndicator.cleanData(ds);
		double[][] profits = new double[num][len - 1];
		for (int i = 0; i < num; i++) {
			profits[i] = FundProfit.fundProfitRatioArray(tmp_datas[i]);
		}
		double[][] datas = new Matrix(profits).transpose().getArrayCopy();
		
		double[][] d_cov = COV.cov(datas);
		double[] d_returns = new double[profits.length];
		for(int i = 0; i < profits.length; i++){
			double[] profit = profits[i];
			double sum = 0.0;
			for(int j = 0; j < profit.length; j++){
				sum = sum + profit[j];
			}
			double avg = sum / profit.length;
			d_returns[i] = avg;
		}
		OUT.printStdout(d_returns);
		
		PCA pca = new PCA(datas);
		double[][] pca_datas = pca.getPc_Array();
		
		double[][] pca_cov = COV.cov(new Matrix(pca_datas).transpose().getArrayCopy());
		double[] pca_returns = new double[pca_datas.length];
		for(int i = 0; i < pca_datas.length; i++){
			double[] pca_data = pca_datas[i];
			double sum = 0.0;
			for(int j = 0; j < pca_data.length; j++){
				sum = sum + pca_data[j];
			}
			double avg = sum / pca_data.length;
			pca_returns[i] = avg;
		}
		
		OUT.printStdout(pca_returns);
//		OUT.printStdout(pca.getPca_vector_matrix().transpose().times(new Matrix(new double[][]{d_returns}).transpose()).getArrayCopy());

//		OUT.printStdout(pca_returns);
		
		double target_return = 0.0007;
//		
		System.out.println("markowitz started");
		Markowitz dmarkowitz = new Markowitz(d_returns, d_cov);
//		OUT.printStdout(d_returns);
		FrontierPoint dfp = dmarkowitz.targetReturn(target_return);
//		System.out.println(dfp);
		double[][] ws = new double[][]{dfp.getWeights()};
//		OUT.printStdout(new Matrix(ws).times(pca.getPca_vector_matrix()).getArrayCopy());
		
		Markowitz pca_markowitz = new Markowitz(pca_returns, pca_cov);
//		OUT.printStdout(pca_returns);
//		OUT.printStdout(pca_cov);
		FrontierPoint pca_fp = pca_markowitz.pca_targetReturn(target_return, pca.getPca_vector_matrix().getArrayCopy());
//		System.out.println(pca_fp);
//		System.out.println("markowitz done");
		double[] weights = pca_fp.getWeights();
		ws = new double[][]{weights};
		OUT.printStdout(ws);
//		OUT.printStdout(new Matrix(ws).times(pca.getPca_vector_matrix().transpose()).getArrayCopy());
		
//		System.out.println(pca_fp);
//		OUT.printStdout(new Matrix(ws).times(pca.getPca_vector_matrix().inverse()).getArray());
//		OUT.printStdout(pca.getPca_vector_matrix().getArray());
//		OUT.printStdout(pca.getPca_vector_matrix().transpose().getArray());
		
		Matrix real_weights = new Matrix(ws).times(pca.getPca_vector_matrix().transpose());
		
		Matrix real_value = real_weights.times(new Matrix(datas).transpose());
		
//		OUT.printStdout(real_value.getArrayCopy());
		double[][] cov = COV.cov(real_value.transpose().getArrayCopy());
//		System.out.println(Math.sqrt(cov[0][0]));
		
	}
	
	
	//@Test
	public void testPCAWeights() throws SQLException, ParseException, FileNotFoundException, UnsupportedEncodingException{
		
		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();
		
		int num = ds.size();
		int len = ds.get(0).size();
		double[][] tmp_datas = FundsIndicator.cleanData(ds);
		double[][] profits = new double[num][len - 1];
		for (int i = 0; i < num; i++) {
			profits[i] = FundProfit.fundProfitRatioArray(tmp_datas[i]);
		}
		double[][] datas = new Matrix(profits).transpose().getArrayCopy();
		
		
		PCA pca = new PCA(datas);
		double[][] pca_datas = pca.getPc_Array();
		
		
		double[][] pca_cov = COV.cov(new Matrix(pca_datas).transpose().getArrayCopy());
		double[] pca_returns = new double[pca_datas.length];
		for(int i = 0; i < pca_datas.length; i++){
			double[] pca_data = pca_datas[i];
			double sum = 0.0;
			for(int j = 0; j < pca_data.length; j++){
				sum = sum + pca_data[j];
			}
			double avg = sum / pca_data.length;
			pca_returns[i] = avg;
		}
		
		OUT.printStdout(pca_returns);
		
		double[] d_returns = new double[profits.length];
		for(int i = 0; i < profits.length; i++){
			double[] profit = profits[i];
			double sum = 0.0;
			for(int j = 0; j < profit.length; j++){
				sum = sum + profit[j];
			}
			double avg = sum / profit.length;
			d_returns[i] = avg;
		}
		OUT.printStdout(d_returns);
		System.out.println("-----------------");
		
		for(int i = 0; i < d_returns.length; i++){
			double r = d_returns[i];
//			System.out.println(r);
			Markowitz pca_markowitz = new Markowitz(pca_returns, pca_cov);
			FrontierPoint pca_fp = pca_markowitz.pca_targetReturn(r, pca.getPca_vector_matrix().getArrayCopy());
	//		System.out.println(pca_fp);
	//		System.out.println("markowitz done");
			double[] weights = pca_fp.getWeights();
//			OUT.printStdout(weights);
			Matrix real_weights = new Matrix(new double[][]{weights}).times(pca.getPca_vector_matrix().transpose());
//			OUT.printStdout(real_weights.getArrayCopy());
		}
	}
	
	@Test
	public void testMinRisk() throws SQLException, ParseException, FileNotFoundException, UnsupportedEncodingException{
		
		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();
		
		int num = ds.size();
		int len = ds.get(0).size();
		double[][] tmp_datas = FundsIndicator.cleanData(ds);
		double[][] profits = new double[num][len - 1];
		for (int i = 0; i < num; i++) {
			profits[i] = FundProfit.fundProfitRatioArray(tmp_datas[i]);
		}
		double[][] datas = new Matrix(profits).transpose().getArrayCopy();
		
		PCA pca = new PCA(datas);
		double[][] cov = pca.getEigen_diagonal().getArrayCopy();
		final String asset_name = "Asset-";
		int camp_length = cov.length;	
		Variable[] tmpVariables = new Variable[camp_length];
		for (int i = 0; i < tmpVariables.length; i++) {
			tmpVariables[i] = (new Variable(asset_name + String.valueOf(i)));
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
				variable_express.setQuadraticFactor(j, i, cov[j][i]);
			}
		}
		variable_express.weight(ONE);

		Optimisation.Result tmpResult = ebm.minimise();
		if (tmpResult.getState().isOptimal() && tmpResult.getState().isSuccess()) {
//			double camp_sd = Math.sqrt(tmpResult.getValue());
			double[] ws = new double[(int) tmpResult.count()];
			for (int n = 0; n < tmpResult.count(); n++) {
				ws[n] = tmpResult.doubleValue(n);
			}
			OUT.printStdout(ws);
			System.out.println(tmpResult.getValue());
			
			Matrix wm = new Matrix(new double[][]{ws});
			Matrix re = wm.times(new Matrix(cov)).times(wm.transpose());
			OUT.printStdout(re.getArrayCopy());
		}
	}
	
	
}
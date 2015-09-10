package com.lcmf.rec.funds.pca;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.List;
import org.apache.log4j.PropertyConfigurator;
import org.junit.Test;

import com.lcmf.rec.funds.indicator.FundProfit;
import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.io.OUT;
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
	
	@Test
	public void testETFPCA() throws SQLException, ParseException, FileNotFoundException, UnsupportedEncodingException{
		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();
		
		int num = ds.size();
		int len = ds.get(0).size();
		double[][] tmp_datas = FundsIndicator.cleanData(ds);
		double[][] profits = new double[num][len-1];
		for(int i = 0; i < num; i++){
			profits[i] = FundProfit.fundProfitRatioArray(tmp_datas[i]);
		}
		double[][] datas = new Matrix(profits).transpose().getArrayCopy();
		
		OUT.printCSV(datas, "./data/tmp/data.csv");
		
		PCA pca = new PCA(datas);
		OUT.printCSV(pca.getPca_vector_matrix().getArray(), "./data/tmp/pc.csv");
		OUT.printCSV(pca.getPc_Array(), "./data/tmp/pca.csv");
	}

}
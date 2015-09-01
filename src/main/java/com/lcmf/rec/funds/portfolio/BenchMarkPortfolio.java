package com.lcmf.rec.funds.portfolio;

import java.sql.SQLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;

import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.markowitz.FrontierPoint;
import com.lcmf.rec.io.db.FundValueReader;

public class BenchMarkPortfolio {

	private static final String hs_300_id = "30000858";
	private static final String zz_500_id = "30000621";
	private static final String js_money_id = "30003314";

	private FundPortfolio hs_300 = null;

	private FundPortfolio zz_500 = null;

	private FundPortfolio js_money = null;

	private static BenchMarkPortfolio bmp = null;

	public static FundPortfolio getBenchMarkPortfolio(String benchMarkName) {
		if (null == bmp) {
			bmp = new BenchMarkPortfolio();
		}
		if ("hs300".equalsIgnoreCase(benchMarkName)) {
			return bmp.hs_300;
		} else if ("zz500".equalsIgnoreCase(benchMarkName)) {
			return bmp.zz_500;
		} else if ("jsmoney".equalsIgnoreCase(benchMarkName)) {
			return bmp.js_money;
		} else {
			return null;
		}

	}

	private BenchMarkPortfolio() {
		load_hs_300();
		load_zz_500();
		load_js_money();
	}

	/**
	 *载入沪深300数据 
	 */
	private void load_hs_300() {

		try {
			FundValueReader reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(hs_300_id);
			reader.readFundValues(ConstVarManager.getPerformance_start_date_str(),
					ConstVarManager.getPerformance_end_date_str());
			reader.close();
			List<List<String>> performance_values = new ArrayList<List<String>>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values.add(reader.getFund_value_seq().get(key));
			}

			reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(hs_300_id);
			reader.readFundValues(ConstVarManager.getModel_start_date_str(), ConstVarManager.getModel_end_date_str());
			reader.close();
			List<List<String>> model_values = new ArrayList<List<String>>();
			keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				model_values.add(reader.getFund_value_seq().get(key));
			}

			FundsIndicator fi = new FundsIndicator(model_values);
			FrontierPoint fp = new FrontierPoint(fi.getReturns()[0], Math.sqrt(fi.getVariance()[0]),
					new double[] { 1.0 }, 0.7);
			FundPortfolio fPortfolio = new FundPortfolio(fp, "hs300", performance_values);
			hs_300 = fPortfolio;

			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "沪深_300", today_str);
			hs_300.setFp_name(fp_name_str);
			hs_300.setRiskvsreturn("风险高，收益低");
			hs_300.setType("hs300");
			

		} catch (SQLException e) {
			e.printStackTrace();
		} catch (ParseException e){
			e.printStackTrace();
		}

	}

	private void load_zz_500() {

		try {
			FundValueReader reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(zz_500_id);
			reader.readFundValues(ConstVarManager.getPerformance_start_date_str(),
					ConstVarManager.getPerformance_end_date_str());
			reader.close();
			List<List<String>> performance_values = new ArrayList<List<String>>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values.add(reader.getFund_value_seq().get(key));
			}

			reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(zz_500_id);
			reader.readFundValues(ConstVarManager.getModel_start_date_str(), ConstVarManager.getModel_end_date_str());
			reader.close();
			List<List<String>> model_values = new ArrayList<List<String>>();
			keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				model_values.add(reader.getFund_value_seq().get(key));
			}

			FundsIndicator fi = new FundsIndicator(model_values);
			FrontierPoint fp = new FrontierPoint(fi.getReturns()[0], Math.sqrt(fi.getVariance()[0]),
					new double[] { 1.0 }, 1.0);
			FundPortfolio fPortfolio = new FundPortfolio(fp, "zz300", performance_values);
			zz_500 = fPortfolio;

			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "中证_500", today_str);
			zz_500.setFp_name(fp_name_str);
			zz_500.setType("zz500");

		} catch (SQLException e) {
			e.printStackTrace();
		} catch (ParseException e){
			e.printStackTrace();
		}

	}

	private void load_js_money() {

		try {
			FundValueReader reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(js_money_id);
			reader.readFundValues(ConstVarManager.getPerformance_start_date_str(),
					ConstVarManager.getPerformance_end_date_str());
			reader.close();
			List<List<String>> performance_values = new ArrayList<List<String>>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values.add(reader.getFund_value_seq().get(key));
			}

			reader = new FundValueReader();
			reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database,
					FundValueReader.username, FundValueReader.password);
			reader.addFundId(js_money_id);
			reader.readFundValues(ConstVarManager.getModel_start_date_str(), ConstVarManager.getModel_end_date_str());
			reader.close();
			List<List<String>> model_values = new ArrayList<List<String>>();
			keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				model_values.add(reader.getFund_value_seq().get(key));
			}

			FundsIndicator fi = new FundsIndicator(model_values);
			FrontierPoint fp = new FrontierPoint(fi.getReturns()[0], Math.sqrt(fi.getVariance()[0]),
					new double[] { 1.0 }, 0.0);
			FundPortfolio fPortfolio = new FundPortfolio(fp, "jsmoney", performance_values);
			js_money = fPortfolio;

			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "嘉实货币A", today_str);
			js_money.setFp_name(fp_name_str);
			js_money.setType("money");

		} catch (SQLException e) {
			e.printStackTrace();
		} catch (ParseException e){
			e.printStackTrace();
		}

	}

	public static String getHs300Id() {
		return hs_300_id;
	}

	public static String getZz500Id() {
		return zz_500_id;
	}

	public static String getJsMoneyId() {
		return js_money_id;
	}
	
	
	
}
package com.lcmf.rec.funds.portfolio;

import java.sql.SQLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;
import org.apache.log4j.Logger;
import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.io.db.FundValueReader;

public class BenchMarkPortfolio {

	private static Logger logger = Logger.getLogger(BenchMarkPortfolio.class);
	
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
		logger.info("load hs_300 data done");
		load_zz_500();
		logger.info("load zz_500 data done");
		load_js_money();
		logger.info("load js_money data done");
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
			List<String> performance_values = new ArrayList<String>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values = reader.getFund_value_seq().get(key);
			}
			
			List<Double> vs = new ArrayList<Double>();
			for(int i = 0; i < performance_values.size(); i++){
				String vstr = performance_values.get(i);
				if(vstr == ""){
					vstr = performance_values.get(i - 1);
				}
				Double v = Double.parseDouble(vstr);
				if(v == 0.0){
					vstr = performance_values.get(i - 1);
					v = Double.parseDouble(vstr);
				}
				vs.add(v);
			}
			

			double[] dvs = new double[vs.size()];
			for(int i = 0; i < vs.size(); i++){
				dvs[i] = vs.get(i);
			}
			hs_300 = new FundPortfolio("hs_300", dvs);
			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "沪深_300", today_str);
			hs_300.setRisk_name(fp_name_str);
			hs_300.setRiskvsreturn("风险高，收益低");
			hs_300.setType("hs300");
			hs_300.computeAnnualReturns();
			hs_300.setRisk(7.0);
			
			
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
			List<String> performance_values = new ArrayList<String>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values = reader.getFund_value_seq().get(key);
			}
			
			List<Double> vs = new ArrayList<Double>();
			for(int i = 0; i < performance_values.size(); i++){
				String vstr = performance_values.get(i);
				if(vstr == ""){
					vstr = performance_values.get(i - 1);
				}
				Double v = Double.parseDouble(vstr);
				if(v == 0.0){
					vstr = performance_values.get(i - 1);
					v = Double.parseDouble(vstr);
				}
				vs.add(v);
			}
			
			double[] dvs = new double[vs.size()];
			for(int i = 0; i < vs.size(); i++){
				dvs[i] = vs.get(i);
			}
			
			zz_500 = new FundPortfolio("zz_500", dvs);
			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "中证_500", today_str);
			zz_500.setRisk_name(fp_name_str);
			zz_500.setType("zz500");
			zz_500.computeAnnualReturns();
			zz_500.setRisk(10.0);

			
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
			List<String> performance_values = new ArrayList<String>();
			Set<String> keys = reader.getFund_value_seq().keySet();
			for (String key : keys) {
				performance_values = reader.getFund_value_seq().get(key);
			}
			
			List<Double> vs = new ArrayList<Double>();
			for(int i = 0; i < performance_values.size(); i++){
				String vstr = performance_values.get(i);
				if(vstr == ""){
					vstr = performance_values.get(i - 1);
				}
				Double v = Double.parseDouble(vstr);
				if(v == 0.0){
					vstr = performance_values.get(i - 1);
					v = Double.parseDouble(vstr);
				}
				vs.add(v);
			}

			
			double[] dvs = new double[vs.size()];
			for(int i = 0; i < vs.size(); i++){
				dvs[i] = vs.get(i);
			}
			
			js_money = new FundPortfolio("js_money", dvs);
			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
			String today_str = format.format(new Date());
			String fp_name_str = String.format("%s_%s", "易方达天天理财货币A", today_str);
			js_money.setRisk_name(fp_name_str);
			js_money.setType("money");
			js_money.computeAnnualReturns();
			js_money.setRisk(0.0);
//
//			SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
//			String today_str = format.format(new Date());
//			String fp_name_str = String.format("%s_%s", "嘉实货币A", today_str);
//			js_money.setFp_name(fp_name_str);
//			js_money.setType("money");
////			js_money.money_values = performance_values.get(0);

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
	
	public static void main(String[] args){
		BenchMarkPortfolio.getBenchMarkPortfolio("hs300");
	}
	
}
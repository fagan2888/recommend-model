package com.lcmf.rec.io.db;

import java.math.BigDecimal;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import org.apache.log4j.Logger;
import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.funds.GlobalVarManager;
import com.lcmf.rec.funds.copywrite.CopyRighter;
import com.lcmf.rec.funds.markowitz.EfficientFrontier;
import com.lcmf.rec.funds.markowitz.FrontierPoint;
import com.lcmf.rec.funds.portfolio.BenchMarkPortfolio;
import com.lcmf.rec.funds.portfolio.FundPortfolio;
import com.lcmf.rec.funds.utils.DateStrList;
import net.sf.json.JSONArray;
import net.sf.json.JSONObject;

public class FundPortfolioMySQLWriter {

	private static Logger logger = Logger.getLogger(FundPortfolioMySQLWriter.class);
	
	RecommendMySQL mysql = new RecommendMySQL();

	public FundPortfolioMySQLWriter() {
		mysql.connect(RecommendMySQL.host, RecommendMySQL.port, RecommendMySQL.database, RecommendMySQL.username,
				RecommendMySQL.password);
	}

	public boolean writeFundPortfolio(FundPortfolio fpf) {
		try {

			/** 计算配置有效日期，后天 */
			Date date = new Date();
			Timestamp tt = new Timestamp(date.getTime());
			Calendar after_tomorrow = Calendar.getInstance();
			after_tomorrow.add(Calendar.DATE, +2);
			Date after_tommmorrow_date = after_tomorrow.getTime();
			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
			String after_tommmorrow_str = format.format(after_tommmorrow_date);

			String sql_base = "replace into fund_portfolios (fp_risk_grade, fp_type, fp_name, fp_annual_return, fp_max_drawdown, fp_risk_vs_return, fp_industry_dispersion, fp_liquidity, fp_expect_return_min, fp_expect_return_max, fp_expired_date, created_at, updated_at) values ('%f','%s','%s','%f','%f', '%s', '%s', '%s', '%f', '%f', '%s','%s', '%s')";
			String sql = String.format(sql_base, fpf.getRisk(), fpf.getType(), fpf.getRisk_name(),
					fpf.getAnnual_return_ratio(), fpf.getMaxDrawdonw(), fpf.getRiskvsreturn(), 1, "高",
					fpf.getExpect_returns_min(), fpf.getExpect_returns_max(), after_tommmorrow_str, tt.toString(),
					tt.toString());
			
			logger.debug(sql);
			mysql.insertDB(sql);

			sql = "select id from fund_portfolios where fp_name = '" + fpf.getRisk_name() + "'";
			ResultSet rs = mysql.selectDB(sql);
			int fp_id = -1;
			if (rs.next()) {
				fp_id = rs.getInt(1);
			} else {
				return false;
			}

			if ("etf".equalsIgnoreCase(fpf.getType())) {
				sql_base = "replace into fund_portfolio_weights (fp_fund_portfolio_id, fp_fund_portfolio_name, fp_fund_id, fp_weight, fp_risk_grade, created_at, updated_at) values ('%d','%s', '%d', '%f', '%f' , '%s', '%s')";
				double[] weights = fpf.getWeights();
				double sum_w = 0.0;
				List<String> fund_codes = fpf.getMofang_ids();
				for (int i = 0; i < fund_codes.size(); i++) {
					//保留两位小数，四舍五入
					double tmp_w = weights[i];
					BigDecimal b = new BigDecimal(tmp_w);
					double w = b.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue();
					if (w <= 0.0)
						continue;
					
					sql = String.format(sql_base, fp_id, fpf.getRisk_name(), Long.parseLong(fund_codes.get(i)), w,
							fpf.getRisk(), tt.toString(), tt.toString());
					logger.debug(sql);
					mysql.insertDB(sql);
					sum_w += w;
				}
			}
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean writeFundProtfolioRiskGrade(FundPortfolio fpf) {
		String header_desc = CopyRighter.risk_grade_header_desc(fpf.getRisk(), 0.7);
		String bottom_desc = CopyRighter.risk_grade_bottom_desc();

		try {
			String sql = "select id from fund_portfolios where fp_name = '" + fpf.getRisk_name() + "'";
			ResultSet rs = mysql.selectDB(sql);
			int fp_id = -1;
			if (rs.next()) {
				fp_id = rs.getInt(1);
			} else {
				return false;
			}

			String sql_base = "replace into fund_portfolio_risks (fpr_risk_grade, fpr_fund_portfolio_id, fpr_header_desc, fpr_bottom_desc, fpr_no_risk_risk, fpr_gold_etf_risk, fpr_hushen300_risk, fpr_zhongzheng500_risk ,created_at, updated_at) values ('%f','%d','%s','%s', '%f','%f','%f','%f', '%s', '%s')";
			Date date = new Date();
			Timestamp tt = new Timestamp(date.getTime());
			sql = String.format(sql_base, fpf.getRisk(), fp_id, header_desc, bottom_desc, 0.0, 0.3, 0.70, 1.0,
					tt.toString(), tt.toString());
			logger.debug(sql);
			mysql.insertDB(sql);

		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public boolean writeFundPortfolioHistory(FundPortfolio fpf) {

		FundPortfolio hs_300 = BenchMarkPortfolio.getBenchMarkPortfolio("hs300");
		String header_desc = "";
		String bottom_desc = "";
		if (null != hs_300) {
			header_desc = CopyRighter.history_header_desc(fpf.getAnnual_return_ratio(),
					hs_300.getAnnual_return_ratio());
			bottom_desc = CopyRighter.history_bottom_desc(fpf.getAnnual_return_ratio(), hs_300.getAnnual_return_ratio(),
					fpf.getMaxDrawdonw(), hs_300.getMaxDrawdonw());
		}

		try {
			String sql = "select id from fund_portfolios where fp_name = '" + fpf.getRisk_name() + "'";
			ResultSet rs = mysql.selectDB(sql);
			int fp_id = -1;
			if (rs.next()) {
				fp_id = rs.getInt(1);
			} else {
				return false;
			}

			List<String> date_str_list = GlobalVarManager.getInstance().getDates_str_list();
			String sql_base = "replace into fund_portfolio_histories (fph_risk_grade, fph_fund_portfolio_id, fph_header_desc, fph_bottom_desc, fph_accumulate_income_rate, fph_points ,created_at, updated_at) values ('%f','%d','%s','%s','%f', '%s', '%s', '%s')";
			Date date = new Date();
			Timestamp tt = new Timestamp(date.getTime());
			sql = String.format(sql_base, fpf.getRisk(), fp_id, header_desc, bottom_desc,
					fpf.getTotal_return_ratio(), historyJson(fpf.getFpValues(), date_str_list), tt.toString(),
					tt.toString());
			logger.debug(sql);
			mysql.insertDB(sql);

		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}

		return true;
	}

	/**
	 * 计算历史表现，转换成json
	 * 
	 * @param values
	 * @param date_str_list
	 * @return
	 */
	public static String historyJson(double[] values, List<String> date_str_list) {
		JSONObject jsonObject = new JSONObject();
		
//		logger.info(values.length);
//		logger.info(date_str_list.size());

		int len = values.length;
		int i = 0;
		int interval = len / 10;
		while (i < len) {
			String date_str = date_str_list.get(i);
			double v = values[i];
			jsonObject.put(date_str, v / values[0] - 1);
			i = i + interval;
		}
		return jsonObject.toString();
	}

	public boolean writeFundProtfolioliquidity(FundPortfolio fpf) {
		String header_desc = CopyRighter.liquidity_header_desc();
		String bottom_desc = CopyRighter.liquidity_bottom_desc();
		try {
			String sql = "select id from fund_portfolios where fp_name = '" + fpf.getRisk_name() + "'";
			ResultSet rs = mysql.selectDB(sql);
			int fp_id = -1;
			if (rs.next()) {
				fp_id = rs.getInt(1);
			} else {
				return false;
			}

			String sql_base = "replace into fund_portfolio_liquidities (fpl_fund_portfolio_id, fpl_header_desc, fpl_bottom_desc, created_at, updated_at) values ('%d','%s', '%s', '%s', '%s')";
			Date date = new Date();
			Timestamp tt = new Timestamp(date.getTime());
			sql = String.format(sql_base, fp_id, header_desc, bottom_desc, tt.toString(), tt.toString());
			logger.debug(sql);
			mysql.insertDB(sql);

		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public boolean writeFundProtfolioExpectTrends(FundPortfolio fp) {
		FundPortfolio hs_300 = BenchMarkPortfolio.getBenchMarkPortfolio("hs300");
		String header_desc = CopyRighter.expect_trends_header_desc(fp.getStd(), hs_300.getU());
		String bottom_desc = CopyRighter.expect_trends_bottom_desc(fp.getExpect_returns_max(), fp.getExpect_returns_min(),
				hs_300.getExpect_returns_max(), hs_300.getExpect_returns_min(), fp.getAnnual_return_ratio(), hs_300.getAnnual_return_ratio());

		try {

			String sql = "select id from fund_portfolios where fp_name = '" + fp.getRisk_name() + "'";
			ResultSet rs = mysql.selectDB(sql);
			int fp_id = -1;
			if (rs.next()) {
				fp_id = rs.getInt(1);
			} else {
				return false;
			}

			Date date = new Date();
			Timestamp tt = new Timestamp(date.getTime());

//			System.out.println(fp.getRisk_name());
//			System.out.println(fp_id);
//			System.out.println(fp.getU());
//			System.out.println(fp.getStd());
//			System.out.println(expectTrendJSON(fp.getU(), fp.getStd()));
//			System.out.println(header_desc);
//			System.out.println(bottom_desc);
//			System.out.println();
//			System.out.println();
//			System.out.println();
//			System.out.println();
			String sql_base = "replace into fund_portfolio_expect_trends (fpe_risk_grade, fpe_fund_portfolio_id, fpe_points, fpe_header_desc, fpe_bottom_desc ,created_at, updated_at) values ('%f','%d','%s','%s','%s','%s','%s')";
			sql = String.format(sql_base, fp.getRisk(), fp_id, expectTrendJSON(fp.getU(), fp.getStd()),
					header_desc, bottom_desc, tt.toString(), tt.toString());
			logger.debug(sql);
			mysql.insertDB(sql);
			return true;
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
	}

	public static String expectTrendJSON(double u, double sigma) {

		List<Double> pt = new ArrayList<Double>();
		List<Double> upper_pt = new ArrayList<Double>();
		List<Double> bottom_pt = new ArrayList<Double>();

		pt.add(1.0);
		upper_pt.add(1.0);
		bottom_pt.add(1.0);
		for (int i = 1; i < 250 * 5; i++) {
			double tmp_p = Math.pow(Math.E, u * i);
			double tmp_upper_p = tmp_p * (1 + sigma * Math.sqrt(i));
			double tmp_bottom_p = tmp_p * (1 - sigma * Math.sqrt(i));
			pt.add(tmp_p);
			upper_pt.add(tmp_upper_p);
			bottom_pt.add(tmp_bottom_p);
		}

		JSONArray jsonArray = new JSONArray();
		int len = pt.size();
		int i = 0;
		while (i < len) {
			JSONArray array = new JSONArray();
			array.add(upper_pt.get(i) - 1);
			array.add(pt.get(i) - 1);
			array.add(bottom_pt.get(i) - 1);
			jsonArray.add(array);
			i = i + 25;
		}
		return jsonArray.toString();

	}

	public void close() {
		mysql.close();
	}

	public static void main(String[] args) {

	}

}

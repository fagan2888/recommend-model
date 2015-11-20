package com.lcmf.rec.io.db;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;
import com.lcmf.rec.funds.portfolio.FundPortfolio;

public class LastPortfolios {

	private static Logger logger = Logger.getLogger(LastPortfolios.class);

	RecommendMySQL mysql = new RecommendMySQL();

	public LastPortfolios() {
		mysql.connect(RecommendMySQL.host, RecommendMySQL.port, RecommendMySQL.database, RecommendMySQL.username,
				RecommendMySQL.password);
	}

	public List<FundPortfolio> getLastPortfolios() throws SQLException {

		String last_portfolio_date_sql = "select pf_date from portfolios order by pf_date desc limit 0, 1";

		ResultSet rs = mysql.selectDB(last_portfolio_date_sql);

		String date = "";
		if (rs.next()) {
			date = rs.getString(1);
		}

		List<String> risk_names = new ArrayList<String>();

		for (int i = 1; i <= 10; i++) {
			risk_names.add("risk" + i + "_" + date);
		}

		List<FundPortfolio> portfolios = new ArrayList<FundPortfolio>();

		int risk = 0;
		for (String name : risk_names) {

			risk++;

			// 从数据库中取出各个基金的weight
			String portfolio_weight_sql = "select pw_fund_id, pw_weight from portfolio_weights where pw_portfolio_name = '"
					+ name + "'";

			rs = mysql.selectDB(portfolio_weight_sql);

			List<String> fids = new ArrayList<String>();
			List<Double> ws = new ArrayList<Double>();

			while (rs.next()) {
				fids.add(String.valueOf(rs.getLong(1)));
				ws.add(rs.getDouble(2));
			}

			// 从数据库中取出该风险基金组合的净值走势
			List<Double> values = new ArrayList<Double>();

			String portfolio_value_sql = String.format(
					"select pv_value from portfolio_values where pv_risk = %f order by pv_date asc", 1.0 * risk / 10);

			rs = mysql.selectDB(portfolio_value_sql);
			while (rs.next()) {
				values.add(rs.getDouble(1));
			}

			double[] vs = new double[values.size()];
			for (int i = 0; i < values.size(); i++) {
				vs[i] = values.get(i);
			}

			// 从数据库中取出该风险基金组合的相关参数
			double annual_returns = 0;
			double expect_returns_max = 0;
			double expect_returns_min = 0;
			String portfolio_sql = String.format(
					"select pf_annual_returns, pf_expect_returns_max, pf_expect_returns_min from portfolios where p_name = '%s'",
					name);
			rs = mysql.selectDB(portfolio_sql);
			if (rs.next()) {
				annual_returns = rs.getDouble(1);
				expect_returns_max = rs.getDouble(2);
				expect_returns_min = rs.getDouble(3);
			}

			for (int j = 0; j <= 9; j++) {
				FundPortfolio portfolio = new FundPortfolio(name, vs);
				portfolio.setRisk(1.0 * risk - 1.0 * j / 10.0);
				portfolio.setMofang_ids(fids);
				portfolio.setWeights(ws);
				portfolio.setAnnual_return_ratio(annual_returns);
				portfolio.setExpect_returns_max(expect_returns_max);
				portfolio.setExpect_returns_min(expect_returns_min);
				portfolio.generateRiskName();
				
				portfolios.add(portfolio);
			}

		}

		mysql.close();
		return portfolios;

	}

	public static void main(String[] args) throws SQLException {
		LastPortfolios lastportfolios = new LastPortfolios();
		List<FundPortfolio> portfolios = lastportfolios.getLastPortfolios();

	}

}
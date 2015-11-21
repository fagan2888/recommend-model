package com.lcmf.rec.offline_fundportfolio_v2;

import java.io.FileNotFoundException;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import com.lcmf.rec.funds.GlobalVarManager;
import com.lcmf.rec.funds.portfolio.BenchMarkPortfolio;
import com.lcmf.rec.funds.portfolio.FundPortfolio;
import com.lcmf.rec.io.db.FundPortfolioMySQLWriter;
import com.lcmf.rec.io.db.LastPortfolios;

/**
 * Hello world!
 *
 */
public class App {
	
	private static Logger logger = Logger.getLogger(App.class);

	public static void main(String[] args) throws SQLException, ParseException, FileNotFoundException {

		PropertyConfigurator.configure("./conf/log4j.properties");
		
		logger.info("started");

		FundPortfolioMySQLWriter fp_writer = new FundPortfolioMySQLWriter();
		
		FundPortfolio hs_300 = BenchMarkPortfolio.getBenchMarkPortfolio("hs300");
		FundPortfolio zz_500 = BenchMarkPortfolio.getBenchMarkPortfolio("zz500");
		FundPortfolio js_money = BenchMarkPortfolio.getBenchMarkPortfolio("jsmoney");

		fp_writer.writeFundPortfolio(hs_300);
		fp_writer.writeFundPortfolioHistory(hs_300);
		fp_writer.writeFundProtfolioRiskGrade(hs_300);
		fp_writer.writeFundProtfolioliquidity(hs_300);
		fp_writer.writeFundProtfolioExpectTrends(hs_300);
		
		logger.info("write hs_300 to database done");
		fp_writer.writeFundPortfolio(zz_500);
		fp_writer.writeFundPortfolioHistory(zz_500);
		fp_writer.writeFundProtfolioRiskGrade(zz_500);
		fp_writer.writeFundProtfolioliquidity(zz_500);
		fp_writer.writeFundProtfolioExpectTrends(zz_500);
		
		logger.info("write zz_500 to database done");
		
		fp_writer.writeFundPortfolio(js_money);
		fp_writer.writeFundPortfolioHistory(js_money);
		fp_writer.writeFundProtfolioRiskGrade(js_money);
		fp_writer.writeFundProtfolioliquidity(js_money);
		fp_writer.writeFundProtfolioExpectTrends(js_money);
		
		logger.info("write money to database done");
		
		logger.info("Create bench mark portfolio done");

		LastPortfolios lastportfolios = new LastPortfolios();
		
		logger.info("get last portfolios done");
		
		List<FundPortfolio> fPortfolios = lastportfolios.getLastPortfolios();
		
		for (FundPortfolio fpft : fPortfolios) {
			fp_writer.writeFundPortfolio(fpft);
			fp_writer.writeFundPortfolioHistory(fpft);
			fp_writer.writeFundProtfolioRiskGrade(fpft);
			fp_writer.writeFundProtfolioliquidity(fpft);
			fp_writer.writeFundProtfolioExpectTrends(fpft);
		}
		fp_writer.close();
		
		logger.info("write all portfolios into database done");
		
	}
}
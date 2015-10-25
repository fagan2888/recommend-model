package com.lcmf.rec.fund_combination;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.funds.FundsCombination;
import com.lcmf.rec.funds.GlobalVarManager;
import com.lcmf.rec.funds.markowitz.EfficientFrontier;
import com.lcmf.rec.funds.markowitz.FrontierPoint;
import com.lcmf.rec.funds.markowitz.Markowitz;
import com.lcmf.rec.funds.portfolio.BenchMarkPortfolio;
import com.lcmf.rec.funds.portfolio.FundPortfolio;
import com.lcmf.rec.io.db.FundPortfolioMySQLWriter;


public class App {

	private static Logger logger = Logger.getLogger(App.class);
	
	public static void main(String[] args) throws SQLException, ParseException, FileNotFoundException {

		PropertyConfigurator.configure("./conf/log4j.properties");
		
		FundPortfolio hs_300 = BenchMarkPortfolio.getBenchMarkPortfolio("hs300");
		FundPortfolio zz_500 = BenchMarkPortfolio.getBenchMarkPortfolio("zz500");
		FundPortfolio js_money = BenchMarkPortfolio.getBenchMarkPortfolio("jsmoney");
		
		
		logger.info("Create bench mark portfolio done");

		FundPortfolioMySQLWriter fp_writer = new FundPortfolioMySQLWriter();
		fp_writer.writeFundPortfolio(hs_300);
		fp_writer.writeFundPortfolioHistory(hs_300);
		fp_writer.writeFundProtfolioExpectTrends(hs_300);
		fp_writer.writeFundPortfolio(zz_500);
		fp_writer.writeFundPortfolioHistory(zz_500);
		fp_writer.writeFundProtfolioExpectTrends(zz_500);
		fp_writer.writeFundPortfolio(js_money);
		fp_writer.writeFundPortfolioHistory(js_money);
		fp_writer.writeFundProtfolioExpectTrends(js_money);

		GlobalVarManager gManager = GlobalVarManager.getInstance();

		FundsCombination fc = new FundsCombination(gManager.getModel_fund_values(), gManager.getFund_mofang_ids());
		
		List<FundPortfolio> fPortfolios = new ArrayList<FundPortfolio>();
		List<FrontierPoint> fPoints = fc.combinations();

//		FrontierPoint p = Markowitz.perfectShape(fPoints, ConstVarManager.getRf());
//		System.out.println(p.getCamp_return() / p.getCamp_sd());
		
		for(int i = 0; i < fPoints.size(); i++){
			fPortfolios.add(new FundPortfolio(fPoints.get(i), "etf", gManager.getPerformance_fund_values()));
//			System.out.println(fPoints.get(i).getCamp_return() + "\t" + fPoints.get(i).getCamp_sd());
		}
		
		
//		for(int i = 0; i < fPortfolios.size(); i++){
//			FundPortfolio fpf = fPortfolios.get(i);
//			if(0.79 == fpf.getFp().getRisk_grade()){
//				PrintStream ps = new PrintStream("./data/tmp.csv");
//				StringBuilder sb = new StringBuilder();
//				for(double v : fpf.getFpValues()){
//					sb.append(v).append(",");
//				}
//				ps.println(sb.toString());
//				ps.flush();
//				ps.close();
//			}
//		}
		
		
		// 计算有效前沿
		EfficientFrontier frontier = new EfficientFrontier(fc.frontierCurve(), "etf");
		EfficientFrontier line = new EfficientFrontier(fc.lowerShapeLine(), "etf");

		// 基金组合存入数据库
		fp_writer.writeFundProtfolioEfficientFrontier(frontier, line);
		
		for (FundPortfolio fpft : fPortfolios) {
			fp_writer.writeFundPortfolio(fpft);
			fp_writer.writeFundPortfolioHistory(fpft);
			fp_writer.writeFundProtfolioRiskGrade(fpft);
			fp_writer.writeFundProtfolioliquidity(fpft);
			fp_writer.writeFundProtfolioRiskVsReturn(frontier, fpft);
			fp_writer.writeFundProtfolioExpectTrends(fpft);
		}
		fp_writer.close();
	}
}
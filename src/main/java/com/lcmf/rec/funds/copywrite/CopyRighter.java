package com.lcmf.rec.funds.copywrite;

/**
 * 各种页面的文案
 * @author yjiaoneal
 *
 */
public class CopyRighter {

	
	public static String history_header_desc(double annual_return, double hs300_return) {
		if (annual_return > hs300_return) {
			return "参考历史数据，当前为您提供的配置方案比沪深300指数获得了更高的收益。";
		} else {
			return "参考历史数据，当前为您提供的配置方案与沪深300指数相比波动更小，更容易获得稳定收益。";
		}
	}

	
	public static String history_bottom_desc(double annual_return, double hs300_return, double portfolio_downdraw,
			double hs300_downdraw) {
		String desc = "";
		if(annual_return > hs300_return && portfolio_downdraw < hs300_downdraw){
			String desc_base = "上图为配置方案与沪深300指数在过去五年的收益表现走势图。通过图表可知，长期来看，当前为您提供的配置方案在过去五年中获取了比沪深300指数高出%.1f%%的平均年化收益，且当遇到行情剧烈波动时，您当前的配置方案更加稳健，较之沪深300指数能够减少%.1f%%的损失。";
			desc = String.format(desc_base, (annual_return - hs300_return) * 100, (hs300_downdraw - portfolio_downdraw) * 100);
		}else if(annual_return > hs300_return && portfolio_downdraw >= hs300_downdraw){
			String desc_base = "上图为配置方案与沪深300指数在过去五年的收益表现走势图。通过图表可知，长期来看，当前为您提供的配置方案在过去几年中获取了比沪深300指数高出%.1f%%的平均年化收益。同时，该配置方案属于高风险高回报的组合，短期内可能会出现较大波动，长期持有能更好的规避风险。";
			desc = String.format(desc_base, (annual_return - hs300_return) * 100);
		}else if(annual_return <= hs300_return){
			String desc_base = "上图为配置方案与沪深300指数在过去五年的收益表现走势图。通过图表可知，当前为您提供的配置方案的历史收益较沪深300指数略低，但当遇到行情剧烈波动时，该配置方案更加稳健，较之沪深300指数能够减少%.1f%%的损失，短期内，较适合在震荡行情下投资。";
			desc = String.format(desc_base, (hs300_downdraw - portfolio_downdraw) * 100);
		}
		return desc;
	}
	

	public static String expect_trends_header_desc(double portfolio_return, double hs300_return) {
		double portfolio_return_annual = Math.pow((1 + portfolio_return), 250) - 1;
		double hs300_return_annual = Math.pow((1 + hs300_return), 250) - 1;
		if (portfolio_return_annual > hs300_return_annual * 1.1) {
			return "基于历史表现及宏观分析，当前为您提供的配置方案将在未来五年比沪深300有更好的收益预期。";
		} else if (portfolio_return_annual <= hs300_return_annual * 1.1 && portfolio_return_annual >= hs300_return_annual * 0.9) {
			return "基于历史表现及宏观分析，当前为您提供的配置方案将在未来五年比沪深300的收益更加稳健。";
		} else {
			return "基于历史表现及宏观分析，当前为您提供的配置方案将在未来五年的收益预计比较稳定，在市场波动的情况中也能获得稳定回报。";
		}
	}

	public static String expect_trends_bottom_desc(double portfolio_return, double hs300_return, double portfolio_sd, double hs300_sd) {
		
		int days = 250;
		
		double portfolio_p = Math.pow(Math.E, (portfolio_return) * days);
		double portfolio_upper_p = portfolio_p * (1 + portfolio_sd * Math.sqrt(days));
		double portfolio_bottom_p = portfolio_p * (1 - portfolio_sd * Math.sqrt(days));

		
		double hs_p = Math.pow(Math.E, (hs300_return) * days);
		double hs_upper_p = hs_p * (1 + hs300_sd * Math.sqrt(days));
		double hs_bottom_p = hs_p * (1 - hs300_sd * Math.sqrt(days));
		
		
		String base = "未来5年，当前为您提供的配置方案在90%%的置信区间中将获得%.1f%%-%.1f%%的平均年化收益，而同期沪深300指数在90%%的置信区间中将获得%.1f%%-%.1f%%的平均年化收益。从图形中显示的收益变化趋势可以看出，该配置方案将使您在未来的投资中";
		String desc = String.format(base, (portfolio_bottom_p - 1) * 100, (portfolio_upper_p - 1) * 100, (hs_bottom_p - 1) * 100, (hs_upper_p - 1) * 100);
		
		double portfolio_return_annual = Math.pow((1 + portfolio_return), 250) - 1;
		double hs300_return_annual = Math.pow((1 + hs300_return), 250) - 1;

		if (portfolio_return_annual > hs300_return_annual * 1.1) {
			return desc + "跑赢大盘。";
		} else if (portfolio_return_annual <= hs300_return_annual * 1.1 && portfolio_return_annual >= hs300_return_annual * 0.9) {
			return desc + "获得更加稳健的获得市场平均收益。";
		} else {
			return desc + "获取稳定回报。";
		}
	}
	
	public static String risk_grade_header_desc(double risk_grade, double hs300_risk_grade){
		if(risk_grade > hs300_risk_grade)
			return String.format("您的风险评测结果为%.1f，高于沪深300指数%.1f的风险值。", risk_grade * 10, hs300_risk_grade * 10);
		else if(risk_grade == hs300_risk_grade)
			return String.format("您的风险评测结果为%.1f，等于沪深300指数%.1f的风险值。", risk_grade * 10, hs300_risk_grade * 10);
		else
			return String.format("您的风险评测结果为%.1f，低于沪深300指数%.1f的风险值。", risk_grade * 10, hs300_risk_grade * 10);
	}
	
	public static String risk_grade_bottom_desc(){
		return "上图为市场标准情况下，不同类型的投资风险分值。风险0分为无风险投资，如将所有资金投资于一年期银行定期存款。根据以上标准，可以定位您当前承担风险的情况，并根据实际情况进行调整。";
	}
	
	public static String risk_vs_return_header_desc(){
		return "当前为您提供的配置方案参考历史和宏观数据的情况下，为当前风险投资性价比最高的配置。";
	}

	public static String risk_vs_return_bottom_desc(){
		return "图中曲线为参考历史数据情况下，绘制出收益和风险的关系图。其中x轴为波动，Y轴为该波动下的平均日收益。为您提供无风险投资、沪深300指数以及中证500指数与您的配置方案进行对比。曲线上的点表示在同等波动情况下可能获得的最高收益。";
	} 	
	
	public static String liquidity_header_desc(){
		return "当前为您提供的配置方案，均是流动性极高的产品，您可以在短时间内随时取用无需担心流动性的问题。";
	}
	
	public static String liquidity_bottom_desc(double sp_weights){
		String percent = String.format("%.1f%%", (1 - sp_weights) * 100);
		return "当前为您提供的理财配置，均是流动性极高的产品。您可以在短时间内随时取用无需担心流动性的问题。" + "\n" + "您配置的方案中，如果您对资金有临时取用的需求，其中" + percent +  "的金额可以在4个工作日内到账。";
	}
	
}

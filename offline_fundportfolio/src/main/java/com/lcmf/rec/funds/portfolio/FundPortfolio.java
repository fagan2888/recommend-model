package com.lcmf.rec.funds.portfolio;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.log4j.Logger;

import com.lcmf.rec.funds.ConstVarManager;
import com.lcmf.rec.funds.FundsCombination;
import com.lcmf.rec.funds.GlobalVarManager;
import com.lcmf.rec.funds.indicator.FundMaxYield;
import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.markowitz.FrontierPoint;

public class FundPortfolio {

	private static Logger logger = Logger.getLogger(FundPortfolio.class);

	private FundsCombination fc = null;
	/** 计算该资产组合的类 */

	private FrontierPoint fp = null;
	/** 有效前沿的点 */

	private String fp_name = "";
	/** 资产组合名字 */

	private String type = "etf";
	/** 资产组合类型 */

	private double[] fpValues = null;
	/** 资产组合的净值走势 */

	private double maxDrawdonw = 0.0;
	/** 最大回撤 */

	private double total_return_ratio = 0.0;
	/** 累计收益 */

	private double annual_return_ratio = 0.0;
	/** 年化收益 */

	private String riskvsreturn = "风险与收益相匹配";

	public FundPortfolio(FrontierPoint fp, String type, List<List<String>> vlist) {
		this.type = type;
		this.fp = fp;
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String today_str = format.format(new Date());
		String fp_id_str = String.format("%s%03d", today_str, Math.round(this.fp.getRisk_grade() * 100));
		this.fp_name = type + "_" + fp_id_str;

		/** 计算组合的净值 */
		this.fpValues = computePerformanceValues(vlist);
		/** 计算组合的最大回撤 */
		this.computeDrawdown();
		/** 计算年化收益 */
		this.computeAnnualReturnRatio();

		List<List<String>> real_values = new ArrayList<List<String>>();
		List<String> vs = new ArrayList<String>();
		for (int i = 0; i < fpValues.length; i++)
			vs.add(String.valueOf(fpValues[i]));
		real_values.add(vs);
		FundsIndicator fi = new FundsIndicator(real_values);
		this.fp.setCamp_return(fi.getReturns()[0]);
		this.fp.setCamp_sd(Math.sqrt(fi.getVariance()[0]));

	}

	public double getCampSd() {
		return this.fp.getCamp_sd();
	}

	public double getCampReturn() {
		return this.fp.getCamp_return();
	}

	public double getMaxDrawdonw() {
		if (0.0 == this.maxDrawdonw) {
			computeDrawdown();
		}
		return maxDrawdonw;
	}

	/**
	 * 预期5年最大值
	 * 
	 * @return
	 */
	public double expectAnnualReturnMax() {

		double u = this.fp.getCamp_return();
		double sigma = this.fp.getCamp_sd();

		int days = fpValues.length;

		double portfolio_p = Math.pow(Math.E, (u) * days);
		// double portfolio_p = Math.pow((1 + u), days);
		// double portfolio_p = this.annual_return_ratio;
		double portfolio_upper_p = portfolio_p * (1 + sigma * Math.sqrt(days));

		double p = Math.pow(portfolio_upper_p, (1 / (1.0 * days / 365)));
		return p - 1;
	}

	/**
	 * 预期5年最小值
	 * 
	 * @return
	 */
	public double expectAnnualReturnMin() {

		double u = this.fp.getCamp_return();
		double sigma = this.fp.getCamp_sd();

		int days = fpValues.length;

		double portfolio_p = Math.pow(Math.E, (u) * days);
		// double portfolio_p = Math.pow((1 + u), days);
		// double portfolio_p = this.annual_return_ratio;
		double portfolio_bottom_p = portfolio_p * (1 - sigma * Math.sqrt(days));
		double p = Math.pow(portfolio_bottom_p, (1 / (1.0 * days / 365)));
		
		//强行把风险是0的收益下沿改成 1.75%
		if(this.fp.getRisk_grade() <= 0.1){
			return 0.0175;
		}
		
		return p - 1;
	}

	/**
	 * 计算最大回撤
	 */
	public void computeDrawdown() {
		this.maxDrawdonw = FundMaxYield.maxYield(this.fpValues);
	}

	/**
	 * 计算年化收益
	 * 
	 * @return
	 */
	public double computeAnnualReturnRatio() {
		double head = fpValues[0];
		double tail = fpValues[fpValues.length - 1];
		int len = fpValues.length;
		this.annual_return_ratio = Math.pow(tail / head, 1 / (1.0 * len / 365)) - 1;
		return this.annual_return_ratio;
	}

	public double getRiskGrade() {
		return fp.getRisk_grade();
	}

	/**
	 * 计算累计收益
	 * 
	 * @return
	 */
	public double computeTotalReturnRatio() {
		this.total_return_ratio = fpValues[fpValues.length - 1] / fpValues[0] - 1;
		return this.total_return_ratio;
	}

	public double[] computePerformanceValues(List<List<String>> pValues) {

		double[] fpValues = null;
		for (List<String> list : pValues) {
			fpValues = new double[list.size()];
		}
		fpValues[0] = 1.0;

		List<List<String>> tmp_values = new ArrayList<List<String>>();
		for (int i = 0; i < pValues.size(); i++) {
			List<String> tmp = new ArrayList<String>();
			for (String v : pValues.get(i)) {
				tmp.add(v);
			}
			tmp_values.add(tmp);
		}

		double[] weights = this.fp.getWeights();

		double sum_w = 0.0;
		for (int i = 0; i < weights.length; i++) {
			sum_w = sum_w + weights[i];
		}

		double[] ws = null;
		if (sum_w <= 0.99) {

			ws = new double[weights.length + 1];
			for (int i = 0; i < weights.length; i++) {
				ws[i] = weights[i];
			}
			ws[weights.length] = 1 - sum_w;
			tmp_values.add(GlobalVarManager.getInstance().getPerformance_money_values());

		} else {

			ws = new double[weights.length + 1];
			for (int i = 0; i < weights.length; i++) {
				ws[i] = weights[i];
			}

		}

		int len = fpValues.length;
		int num = tmp_values.size();

		for (int i = 1; i < len; i++) {
			double profit = 0.0;
			for (int j = 0; j < num; j++) {
				double w = ws[j];
				String today_value_str = tmp_values.get(j).get(i);
				int m = i - 1;
				if (today_value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(today_value_str)) {
					continue;
				}
				while (m >= 0) {
					String value_str = tmp_values.get(j).get(m);
					if (value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(value_str)) {
						m--;
					} else {
						profit += w * (Double.parseDouble(today_value_str) / Double.parseDouble(value_str) - 1);
						break;
					}
				}
			}
			fpValues[i] = fpValues[i - 1] * (profit + 1);
		}

		return fpValues;
	}

	public double[] getFpValues() {
		return fpValues;
	}

	public double[] getWeights() {
		return this.fp.getWeights();
	}

	public double getTotal_return_ratio() {
		return total_return_ratio;
	}

	public double getAnnual_return_ratio() {
		return annual_return_ratio;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getFp_name() {
		return fp_name;
	}

	public void setFp_name(String fp_name) {
		this.fp_name = fp_name;
	}

	public String getRiskvsreturn() {
		return riskvsreturn;
	}

	public void setRiskvsreturn(String riskvsreturn) {
		this.riskvsreturn = riskvsreturn;
	}

	public FrontierPoint getFp() {
		return fp;
	}

	public static void main(String[] args) {

	}

}

/*package com.lcmf.rec.funds.portfolio;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import com.lcmf.rec.funds.indicator.FundMaxYield;
import com.lcmf.rec.funds.markowitz.FrontierPoint;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;

public class OldFundPortfolio {

	private double rf = 0.0425 / 365;

	private FrontierPoint point = null;

	private double risk_grade = -1;

	private String fp_name = "";

	private double[] weights = null;

	private String type = null;

	private List<String> fund_codes = null;

	private List<String> d_list = null;

	private HashMap<String, List<String>> fund_value_seq = null;

	private double[] values = null;

	private double drawdown = 0.0;

	private double accumulated_income_rate = 0.0;
	
	private String riskvsreturn = "风险与收益相匹配";

	public OldFundPortfolio(FrontierPoint fp, HashMap<String, List<String>> seq, String type) {
		
		this.type = type;
		this.point = fp;
		this.fund_value_seq = seq;
		this.risk_grade = fp.risk_grade;
		for (List<String> list : seq.values()) {
			values = new double[list.size()];
		}
		values[0] = 1.0;
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String today_str = format.format(new Date());
		String fp_id_str = String.format("%s%03d", today_str, Math.round(this.risk_grade * 100));
		this.fp_name = type + "_" + fp_id_str;
		this.fund_codes = new ArrayList<String>();
		for (String key : seq.keySet()) {
			this.fund_codes.add(key);
		}
		Collections.sort(this.fund_codes);
		weights = fp.weights;

	}

	
	public String getRiskvsreturn() {
		return riskvsreturn;
	}

	

	public void setRisk_grade(double risk_grade) {
		this.risk_grade = risk_grade;
	}


	public void setRiskvsreturn(String riskvsreturn) {
		this.riskvsreturn = riskvsreturn;
	}


	public double[] getWeights() {
		return weights;
	}

	public List<String> getFund_codes() {
		return fund_codes;
	}

	public double getDrawdown() {
		this.drawdown = FundMaxYield.maxYield(this.values);
		return drawdown;
	}

	public double getRisk_grade() {
		return risk_grade;
	}

	public String getFp_name() {
		return fp_name;
	}

	public String getType() {
		return type;
	}

	public void setFp_name(String fp_name) {
		this.fp_name = fp_name;
	}

	public double[] getValues() {
		computeValues();
		return values;
	}

	public void setD_list(List<String> d_list) {
		this.d_list = d_list;
	}

	public double getCampSd() {
		return this.point.camp_sd;
	}

	public double getCampReturn() {
		return this.point.camp_return;
	}

	public double getAccumulated_income_rate() {
		computeAccumulatedIncomeRate();
		return accumulated_income_rate;
	}

	*//**
	 * 计算年化收益率
	 *//*
	public void computeAccumulatedIncomeRate() {
		this.accumulated_income_rate = values[values.length - 1] / values[0] - 1;
	}

	*//**
	 * 计算最大回撤
	 *//*
	public void computeDrawdown() {
		this.drawdown = FundMaxYield.maxYield(this.values);
	}

	*//**
	 * 计算历史走势
	 * 
	 * @return
	 *//*
	public String computeHistory() {
		JSONObject jsonObject = new JSONObject();
		int len = values.length;
		int i = 0;
		int interval = len / 100;
		while (i < len) {
			String date_str = d_list.get(i);
			double v = values[i];
			jsonObject.put(date_str, v);
			i = i + interval;
		}
		return jsonObject.toString();
	}

	*//**
	 * 预期年化最大值
	 * @return
	 *//*
	public float expectReturnMax(){
		
		double u = this.point.camp_return;
		double sigma = this.point.camp_sd;
		
		double p = Math.pow(Math.E, u * 250);
		double upper_p = p + Math.pow(Math.E, sigma * Math.sqrt(250));
		return (float)upper_p;
	}
	
	
	*//**
	 * 预期年化最小值
	 * @return
	 *//*
	public float expectReturnMin(){
		
		double u = this.point.camp_return;
		double sigma = this.point.camp_sd;
		
		double p = Math.pow(Math.E, u * 250);
		double bottom_p = p - Math.pow(Math.E, sigma * Math.sqrt(250));
		return (float)bottom_p;
	}
	
	public String computeExpectTrend() {

		double u = this.point.camp_return;
		double sigma = this.point.camp_sd;

		List<Double> pt = new ArrayList<Double>();
		List<Double> upper_pt = new ArrayList<Double>();
		List<Double> bottom_pt = new ArrayList<Double>();

		pt.add(1.0);
		upper_pt.add(1.0);
		bottom_pt.add(1.0);
		for (int i = 1; i < 250 * 5; i++) {
			double tmp_p = Math.pow((1 + u), i);
			double tmp_upper_p = tmp_p * (1 + sigma * Math.sqrt(i));
			double tmp_bottom_p = tmp_p * (1 - sigma * Math.sqrt(i));
			pt.add(tmp_p);
			upper_pt.add(tmp_upper_p);
			bottom_pt.add(tmp_bottom_p);
		}

		JSONArray jsonArray = new JSONArray();
		int len = pt.size();
		int i = 0; 
		while(i < len){
			JSONArray array = new JSONArray();
			array.add(upper_pt.get(i));
			array.add(pt.get(i));
			array.add(bottom_pt.get(i));
			jsonArray.add(array);
			i = i + 25;
		}
		return jsonArray.toString();
	}

	*//**
	 * 年化收益率
	 * 
	 * @return
	 *//*
	public double annual() {
		double head = values[0];
		double tail = values[values.length - 1];
		int len = values.length;
		return Math.pow(tail / head, 1 / (1.0 * len / 365)) - 1;
	}

	*//**
	 * 计算基金净值走势
	 *//*
	public void computeValues() {

		List<List<String>> values = new ArrayList<List<String>>();
		HashMap<String, List<String>> map = this.fund_value_seq;
		List<String> fund_code = new ArrayList<String>();
		for (String key : map.keySet()) {
			fund_code.add(key);
		}

		Collections.sort(fund_code);
		for (int i = 0; i < fund_code.size(); i++) {
			values.add(map.get(fund_code.get(i)));
		}

		double[] weights = this.point.weights;
		int len = this.values.length;
		int num = values.size();

		for (int i = 1; i < len; i++) {
			double profit = 0.0;
			double sum_w = 0.0;
			for (int j = 0; j < num; j++) {
				double w = weights[j];
				sum_w += w;
				String today_value_str = values.get(j).get(i);
				int m = i - 1;
				if (today_value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(today_value_str)) {
					continue;
				}
				while (m >= 0) {
					String value_str = values.get(j).get(m);
					if (value_str.equalsIgnoreCase("") || 0 == Double.parseDouble(value_str)) {
						m--;
					} else {
						profit += w * (Double.parseDouble(today_value_str) / Double.parseDouble(value_str) - 1);
						break;
					}
				}
			}
			profit = profit * sum_w + rf * (1 - sum_w);
			this.values[i] = this.values[i - 1] * (profit + 1);
		}
	}
}
*/
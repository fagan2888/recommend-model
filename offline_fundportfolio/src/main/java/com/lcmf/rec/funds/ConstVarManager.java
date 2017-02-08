package com.lcmf.rec.funds;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class ConstVarManager {
	
	private static String  model_start_date_str = ""; 
	
	private static String  model_end_date_str  = "";
	
	private static String  performance_start_date_str = ""; 
	
	private static String  performance_end_date_str  = "";
	
	private static double rf = 0.030 / 250; /** 无风险收益率*/ 
	
	static {
		model_start_date_str = "2006-01-04";
		model_end_date_str = "2015-05-30";
//		performance_start_date_str = "2009-09-04";
		performance_start_date_str = "2009-02-04";
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
		Calendar calendar = Calendar.getInstance();
		calendar.setTime(new Date());
		int day = calendar.get(Calendar.DATE);
		calendar.set(Calendar.DATE, day - 1);
//		calendar.set(Calendar.DATE, day - 2);
		performance_end_date_str = format.format(calendar.getTime());
//		System.out.println(performance_end_date_str);
	}

	public static double getRf() {
		return rf;
	}
	
	public static String getModel_start_date_str() {
		return model_start_date_str;
	}

	public static String getModel_end_date_str() {
		return model_end_date_str;
	}

	public static String getPerformance_start_date_str() {
		return performance_start_date_str;
	}

	public static String getPerformance_end_date_str() {
		return performance_end_date_str;
	}

	public static void setPerformance_start_date_str(String performance_start_date_str) {
		ConstVarManager.performance_start_date_str = performance_start_date_str;
	}

	public static void setPerformance_end_date_str(String performance_end_date_str) {
		ConstVarManager.performance_end_date_str = performance_end_date_str;
	}

	public static void setModel_start_date_str(String model_start_date_str) {
		ConstVarManager.model_start_date_str = model_start_date_str;
	}

	public static void setModel_end_date_str(String model_end_date_str) {
		ConstVarManager.model_end_date_str = model_end_date_str;
	}

}
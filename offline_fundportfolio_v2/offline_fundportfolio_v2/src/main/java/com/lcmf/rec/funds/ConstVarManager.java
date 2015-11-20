package com.lcmf.rec.funds;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class ConstVarManager {

	private static String  performance_start_date_str = ""; 
	
	private static String  performance_end_date_str  = "";
	
	private static double rf = 0.0175 / 250; /** 无风险收益率*/ 
	
	static {
//		performance_start_date_str = "2009-09-04";
//		performance_start_date_str = "2013-05-30";
		List<String> dates = GlobalVarManager.getInstance().getDates_str_list();
		performance_start_date_str = dates.get(0);
		performance_end_date_str   = dates.get(dates.size() - 1);
//		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
//		Calendar calendar = Calendar.getInstance();
//		calendar.setTime(new Date());
//		int day = calendar.get(Calendar.DATE);
//		calendar.set(Calendar.DATE, day - 1);
////		calendar.set(Calendar.DATE, day - 2);
//		performance_end_date_str = format.format(calendar.getTime());
//		System.out.println(performance_end_date_str);
	}

	public static double getRf() {
		return rf;
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

}

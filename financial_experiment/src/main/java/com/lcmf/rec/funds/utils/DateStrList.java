package com.lcmf.rec.funds.utils;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class DateStrList {

	/**
	 * 返回从开始日期到结束日期中每一天的日期
	 * @param start_date_str
	 * @param end_date_str
	 * @return
	 */
	public static List<String> dList(String start_date_str, String end_date_str){
		ArrayList<String> all_date_str_list = new ArrayList<String>();
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
		
		try {
			Date start_date = format.parse(start_date_str);
			Date end_date = format.parse(end_date_str);
			Calendar c = Calendar.getInstance();
			Date current_date = start_date;
			while (current_date.getTime() <= end_date.getTime()) {
				c.setTime(current_date);
				all_date_str_list.add(format.format(c.getTime()));
				int day = c.get(Calendar.DATE);
				c.set(Calendar.DATE, day + 1);
				current_date = c.getTime();
			}
			return all_date_str_list;
		} catch (ParseException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		List<String> date_list = DateStrList.dList("2009-09-04", "2015-10-13");
		System.out.println(date_list);
		PrintStream ps = new PrintStream("./data/date.csv");
		StringBuilder sb = new StringBuilder();
		for(String v : date_list){
			sb.append(v).append(",");
		}
		ps.println(sb.toString());
		ps.flush();
		ps.close();
	}

}
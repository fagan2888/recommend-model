package com.lcmf.rec.funds.data;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.List;

import com.lcmf.rec.funds.utils.DateStrList;
import com.lcmf.rec.io.db.FundValueReader;

public class FundValuesTrend {

	public static void main(String[] args) {

//		List<String> lines = IOUtils.readLines(new FileInputStream("./data/指数时间轴.csv"),"gbk");
//		String date_str = lines.get(0).trim().replaceAll("/", "-");
//		List<String> dates = new ArrayList<String>();
//		for(String d : date_str.split(","))
//			dates.add(d);
//		dates.remove(0);
//		
//		FundValueReader reader = new FundValueReader();
//		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
//				FundValueReader.password);
//		reader.readFundIds("./data/chunzhai.ids");
//		reader.readFundValues("2001-01-01", "2015-05-30");
//
//		
//		PrintStream ps = new PrintStream("./data/fund_values.csv");
//		StringBuilder sb = new StringBuilder();
//		sb.append(",");
//		for(String date : dates){
//			sb.append(date).append(",");
//		}
//		ps.println(sb.toString());
//		
//		for (String key : reader.fund_values.keySet()) {
//			String fund_id = reader.fund_ids_map.get(key);
//			sb = new StringBuilder();
//			sb.append(fund_id).append(",");
//			Double pre_v = null;
//			HashMap<String, Double> values = reader.fund_values.get(key);
//			for(String d : dates){
//				Double v = values.get(d);
//				if(null == v){
//					if(null == pre_v)
//						sb.append(",");
//					else
//						sb.append(pre_v).append(",");
//				}else if (0 == v){
//					if(null == pre_v)
//						sb.append(",");
//					else
//						sb.append(pre_v).append(",");
//				}else{
//					sb.append(v).append(",");
//					pre_v = v;
//				}
//			}
//			ps.println(sb.toString());
//		}
//		ps.close();
		
//		for (String key : reader.fund_values.keySet()) {
//			System.out.println(key);
//			HashMap<String, Double> values = reader.fund_values.get(key);
//			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
//			List<Date> d_list = new ArrayList<Date>();
//			for (String d_str : values.keySet()) {
//				d_list.add(format.parse(d_str));
//			}
//
//			Collections.sort(d_list);
//			List<String> d_str_list = new ArrayList<String>();
//			for (Date d : d_list) {
//				d_str_list.add(format.format(d));
//			}
//			System.out.println(d_str_list);
//		}
		
//		PrintStream ps = new PrintStream("./data/fund_values.csv");
//		StringBuilder sb = new StringBuilder();
//		sb.append(",");
//		for(String date : reader.getDate_list()){
//			sb.append(date).append(",");
//		}
//		ps.println(sb.toString());
//		
//		HashMap<String, List<String>> map = reader.fund_value_seq;
//		for (String key : map.keySet()) {
//			List<String> list = map.get(key);
//			sb = new StringBuilder();
//			sb.append(key).append(",");
//			for (String str : list) {
//				sb.append(str).append(",");
//			}
//			ps.println(sb.toString());
//		}
//		ps.close();
		
//		FundValueReader reader = new FundValueReader();
//		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
//				FundValueReader.password);
//		reader.readFundIds("./data/chunzhai.ids");
//		reader.readFundValues("2009-09-04", "2015-05-30");
		
		
	}		

}

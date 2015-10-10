package com.lcmf.rec.io.db;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import org.apache.commons.io.IOUtils;

public class FundValueReader {

	private List<String> fund_ids = new ArrayList<String>();

	private List<String> fund_mofang_ids = new ArrayList<String>();
	
	private HashMap<String, String> fund_ids_map = new HashMap<String, String>();
	
	private HashMap<String, List<String>> fund_value_seq = new HashMap<String, List<String>>();
	
	private HashMap<String, HashMap<String, Double>> fund_values = new HashMap<String, HashMap<String, Double>>();

	private List<String> date_list = new ArrayList<String>();
	
	public static String host = "182.92.214.1";

	public static String port = "3306";

	public static String database = "mofang";

	public static String username = "jiaoyang";

	public static String password = "Mofang123";

	private static String DriverName = "com.mysql.jdbc.Driver";

	private static String ConnString = "jdbc:mysql://%s:%s/%s?user=%s&password=%s&useUnicode=true&characterEncoding=utf8&autoReconnect=true";

	static {
		try {
			Properties prop = new Properties();
			FileInputStream fis= new FileInputStream("./conf/mofang.db");
			prop.load(fis);
			host = prop.getProperty("host", "182.92.214.1");
			port = prop.getProperty("port", "3306");
			database = prop.getProperty("database", "mofang");
			username = prop.getProperty("username", "jiaoyang");
			password = prop.getProperty("password", "Mofang123");
			fis.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private Connection conn = null;

	/**
	 * close database connection
	 * 
	 * @return
	 */
	public boolean close() {
		try {
			conn.close();
			return true;
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return false;
	}

	/**
	 * connect to mysql database
	 * 
	 * @return true
	 */
	public boolean connect(String host, String port, String database, String username, String password) {

		String url = String.format(ConnString, host, port, database, username, password);

		try {
			Class.forName(DriverName);
			conn = DriverManager.getConnection(url);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			return false;
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * 
	 * @param sql
	 *            statement for select
	 * @return resultset
	 * @throws SQLException
	 */
	public ResultSet selectDB(String sql) throws SQLException {
		Statement statement = conn.createStatement();
		ResultSet rs = statement.executeQuery(sql);
		return rs;
	}

	public void readFundIds(String path) throws SQLException {
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line = null;
			while (null != (line = br.readLine())) {
				fund_ids.add(line.trim());
				String fi_gid_sql = String.format("select fi_globalid from fund_infos where fi_code=%s", line.trim());
				ResultSet rs = selectDB(fi_gid_sql);
				if (rs.next()) {
					fund_mofang_ids.add(String.valueOf(rs.getInt(1)));
					fund_ids_map.put(String.valueOf(rs.getInt(1)), line.trim());
				}
			}
			Collections.sort(fund_mofang_ids);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void addFundId(String fund_id){
		fund_mofang_ids.add(String.valueOf(fund_id));
		Collections.sort(fund_mofang_ids);
	}
	
	public void readFundValues(String start_date_str, String end_date_str) throws SQLException, ParseException {

		ArrayList<String> all_date_str_list = new ArrayList<String>();
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
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
		
		this.date_list = all_date_str_list;
		
		for (int i = 0; i < fund_mofang_ids.size(); i++) {
			String fi_id = fund_mofang_ids.get(i).trim();
			String select_sql = String
					.format("select fv_time, fv_authority_value from fund_value where fv_fund_id = '%s'", fi_id);
//			System.out.println(select_sql);
			ResultSet rs = selectDB(select_sql);
			HashMap<String, Double> values = new HashMap<String, Double>();
			while (rs.next()) {
				values.put(rs.getString(1), rs.getDouble(2));
			}
			fund_values.put(fi_id, values);
		}

		for(String key : fund_values.keySet()){
			HashMap<String, Double> values = fund_values.get(key);
			ArrayList<String> values_seq = new ArrayList<String>();
			for(int i = 0 ; i < all_date_str_list.size(); i++){
				String date_str = all_date_str_list.get(i);
				Double v = values.get(date_str);
				if(v == null){
					values_seq.add("");
				}else{
					if(0.0 == v){
						values_seq.add("");
					}else{
						values_seq.add(String.valueOf(v));
					}
				}
			}
			fund_value_seq.put(key, values_seq);
		}
	}

	
	
	public HashMap<String, List<String>> getFund_value_seq() {
		return fund_value_seq;
	}

	public HashMap<String, HashMap<String, Double>> getFund_values() {
		return fund_values;
	}

	public List<List<String>> getValueList(){
		List<List<String>> values = new ArrayList<List<String>>();
		for (String key : fund_mofang_ids) {
			values.add(this.getFund_value_seq().get(key));
		}
		return values;
	}
	
	
	public List<String> getDate_list() {
		return date_list;
	}

	
	public HashMap<String, String> getFund_ids_map() {
		return fund_ids_map;
	}

	public void setFund_ids_map(HashMap<String, String> fund_ids_map) {
		this.fund_ids_map = fund_ids_map;
	}

	public static void main(String[] args) throws SQLException, ParseException, IOException {

		List<String> lines = IOUtils.readLines(new FileInputStream("./data/指数时间轴.csv"),"gbk");
		String date_str = lines.get(0).trim().replaceAll("/", "-");
		List<String> dates = new ArrayList<String>();
		for(String d : date_str.split(","))
			dates.add(d);
		dates.remove(0);
		
		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./data/chunzhai.ids");
		reader.readFundValues("2001-01-01", "2015-05-30");

		
		PrintStream ps = new PrintStream("./data/fund_values.csv");
		StringBuilder sb = new StringBuilder();
		sb.append(",");
		for(String date : dates){
			sb.append(date).append(",");
		}
		ps.println(sb.toString());
		
		for (String key : reader.fund_values.keySet()) {
			String fund_id = reader.fund_ids_map.get(key);
			sb = new StringBuilder();
			sb.append(fund_id).append(",");
			Double pre_v = null;
			HashMap<String, Double> values = reader.fund_values.get(key);
			for(String d : dates){
				Double v = values.get(d);
				if(null == v){
					if(null == pre_v)
						sb.append(",");
					else
						sb.append(pre_v).append(",");
				}else{
					sb.append(v).append(",");
					pre_v = v;
				}
			}
			ps.println(sb.toString());
		}
		ps.close();
		
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
		
		
	}
}
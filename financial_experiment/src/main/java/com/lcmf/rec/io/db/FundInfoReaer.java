package com.lcmf.rec.io.db;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;

public class FundInfoReaer {

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

	public Integer fundCode(Integer id) throws SQLException{
		String sql = String.format("select * from fund_infos where fi_globalid = %d", id);
		
		ResultSet rs = this.selectDB(sql);
		
		if(rs.next())
			return rs.getInt("fi_code");
		return null;
	}
	
}

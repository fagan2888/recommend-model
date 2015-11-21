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

import com.jcraft.jsch.Logger;

public class RecommendMySQL {

	public static String host = "dev.mofanglicai.com.cn";

	public static String port = "3306";

	public static String database = "recommend";

	public static String username = "jiaoyang";

	public static String password = "q36wx5Td3Nv3Br2OPpH7";

	private static String DriverName = "com.mysql.jdbc.Driver";

	private static String ConnString = "jdbc:mysql://%s:%s/%s?user=%s&password=%s&useUnicode=true&characterEncoding=utf8&autoReconnect=true";

	private Connection conn = null;
	
	static {
		try {
			Properties prop = new Properties();
			FileInputStream fis = new FileInputStream("./conf/recommend.db");
			prop.load(fis);
			host = prop.getProperty("host", "dev.mofanglicai.com.cn");
			port = prop.getProperty("port", "3306");
			database = prop.getProperty("database", "recommend");
			username = prop.getProperty("username", "jiaoyang");
			password = prop.getProperty("password", "q36wx5Td3Nv3Br2OPpH7");
			fis.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

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
		System.out.println(url);
		
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

	/**
	 * statement for insert
	 * 
	 * @param sql
	 * @return
	 * @throws SQLException
	 */
	public void insertDB(String sql) throws SQLException {
		Statement statement = conn.createStatement();
		statement.executeUpdate(sql);
	}

	public static void main(String[] args) {
		RecommendMySQL mysql = new RecommendMySQL();
		mysql.connect(RecommendMySQL.host, RecommendMySQL.port, RecommendMySQL.database,
				RecommendMySQL.username, RecommendMySQL.password);

	}

}

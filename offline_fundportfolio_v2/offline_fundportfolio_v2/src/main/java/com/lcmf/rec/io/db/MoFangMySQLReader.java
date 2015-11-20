/**
 * 
 */
package com.lcmf.rec.io.db;

import java.io.FileNotFoundException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;


/**
 * Read mofang mysql database
 * @author yjiaoneal
 *
 */
public class MoFangMySQLReader {
	
	public static final String host = "182.92.214.1";

	public static final String port = "3306";

	public static final String database = "recommend";

	public static final String username = "jiaoyang";

	public static final String password = "Mofang123";

	private static final String DriverName = "com.mysql.jdbc.Driver";

	private static final String ConnString = "jdbc:mysql://%s:%s/%s?user=%s&password=%s&useUnicode=true&characterEncoding=utf8&autoReconnect=true";

	private Connection conn = null;

	
	/**
	 * close database connection
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
	 * @return true
	 */
	public boolean connect(String host, String port, String database, String username, String password) {

		String url = String.format(ConnString, host, port, database, username, password);

		// System.out.println(url);

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
	 * @param sql statement for select
	 * @return resultset
	 * @throws SQLException
	 */
	public ResultSet selectDB(String sql) throws SQLException {
		Statement statement = conn.createStatement();
		ResultSet rs = statement.executeQuery(sql);
		return rs;
	}

	/**
	 * @param args
	 * @throws FileNotFoundException
	 * @throws CloneNotSupportedException 
	 */
	public static void main(String[] args) throws FileNotFoundException, CloneNotSupportedException {
		// TODO Auto-generated method stub
		
//		String remoteSSHHost = "182.92.214.1";
//
//		String remoteSSHUser = "jiaoyang";
//
//		String remoteSSHPwd = "jiao123456";

//		int remoteSSHPort = 22;
		
//		String host = "182.92.214.1";
//
//		String port = "3306";
//
//		String database = "mofang";
//
//		String username = "jiaoyang";
//
//		String password = "Mofang123";
		
//		ShellTunnel st = new ShellTunnel();
//
//		st.connectSSH(remoteSSHHost, remoteSSHUser, remoteSSHPwd, remoteSSHPort);
//		st.portForwarding(3306, "127.0.0.1", 3306);
		// System.out.println(connected);

		MoFangMySQLReader reader = new MoFangMySQLReader();
		reader.connect(host, port, database, username, password);

		reader.clone();
		
//		st.disconnect();
	}

}

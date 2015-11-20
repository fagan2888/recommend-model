package com.lcmf.rec.io.db;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.util.Date;
import java.util.List;

import com.lcmf.rec.risk_analyze_question.Question;

public class RiskAnalyzeQuestionMySQL {

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

	public void insertQuestion(Question q) {
		// if(q == null)
		// return;

		Date date = new Date();
		Timestamp tt = new Timestamp(date.getTime());
		String q_sql = "insert into user_risk_analyze_questions(ur_question, ur_q_type, created_at, updated_at) values ('%s', '%s','%s', '%s')";
		String sql = String.format(q_sql, q.question, q.type, tt.toString(), tt.toString());
		try {
			Statement st = conn.createStatement();
			st.execute(sql);
			if (q.type.equalsIgnoreCase("choice")) {
				List<String> ops = q.options;
				String select_sql = "select id from user_risk_analyze_questions where ur_question = '" + q.question
						+ "'";
				System.out.println(select_sql);
				ResultSet rs = this.selectDB(select_sql);
				if (rs.next()) {
					int question_id = rs.getInt(1);
					String option_sql = "insert into user_risk_analyze_question_options (ur_question_id, ur_option_a, ur_option_b, ur_option_c, ur_option_d, ur_option_e, ur_option_f, created_at, updated_at) values ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')";
					sql = String.format(option_sql, question_id, ops.size() >= 1 ? ops.get(0) : "",
							ops.size() >= 2 ? ops.get(1) : "", ops.size() >= 3 ? ops.get(2) : "",
							ops.size() >= 4 ? ops.get(3) : "", ops.size() >= 5 ? ops.get(4) : "",
							ops.size() >= 6 ? ops.get(5) : "", tt.toString(), tt.toString());
					System.out.println(sql);
					st.execute(sql);
				}
			}
		} catch (SQLException e) {
			e.printStackTrace();
		}

	}

	public static void main(String[] args) {

		RiskAnalyzeQuestionMySQL writer = new RiskAnalyzeQuestionMySQL();
		writer.connect(RiskAnalyzeQuestionMySQL.host, RiskAnalyzeQuestionMySQL.port, RiskAnalyzeQuestionMySQL.database,
				RiskAnalyzeQuestionMySQL.username, RiskAnalyzeQuestionMySQL.password);
		Question q = null;
		writer.insertQuestion(q);
	}

}

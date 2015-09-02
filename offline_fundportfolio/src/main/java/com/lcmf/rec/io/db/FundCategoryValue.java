package com.lcmf.rec.io.db;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;


public class FundCategoryValue {
	

	private static final String host = "182.92.214.1";

	private static final String port = "3306";

	private static final String database = "mofang";

	private static final String username = "jiaoyang";

	private static final String password = "Mofang123";
	

	public static void main(String[] args) throws FileNotFoundException {
	//	ShellTunnel st = new ShellTunnel();

	//	st.connectSSH(remoteSSHHost, remoteSSHUser, remoteSSHPwd, remoteSSHPort);
	//	st.portForwarding(3306, "127.0.0.1", 3306);
		// System.out.println(connected);

		MoFangMySQLReader reader = new MoFangMySQLReader();
		reader.connect(host, port, database, username, password);

		// System.out.println(connected);

		ArrayList<Integer> types = new ArrayList<Integer>();
		String sql = "select * from type_infos where ti_mark = \"基金类型\"";
		// PrintStream ps = new PrintStream("./data/tmp/funds_type");
		try {
			ResultSet rs = reader.selectDB(sql);
			while (rs.next()) {
				types.add(rs.getInt(1));
			}
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}

		// ps.close();

		HashMap<Integer, ArrayList<Integer>> type_funds_id = new HashMap<Integer, ArrayList<Integer>>();

		ArrayList<Integer> all_funds_id = new ArrayList<Integer>();

		for (int i = 0; i < types.size(); i++) {
			int type = types.get(i);
			sql = "select * from item_type where it_type_id=" + String.valueOf(type);
			ArrayList<Integer> funds_id = new ArrayList<Integer>();
			try {
				ResultSet rs = reader.selectDB(sql);
				while (rs.next()) {
					funds_id.add(rs.getInt(2));
					all_funds_id.add(rs.getInt(2));
				}
				rs.close();
			} catch (SQLException e) {
				e.printStackTrace();
			}
			type_funds_id.put(type, funds_id);
			// System.out.println(type);
			// System.out.println(funds_id);

		}

		HashMap<Integer, Integer> funds_id_map = new HashMap<Integer, Integer>();

		sql = "select * from fund_infos";
		try {
			ResultSet rs = reader.selectDB(sql);
			while (rs.next()) {
				int fi_globalid = rs.getInt(1);
				int fi_code = rs.getInt(2);
				if (all_funds_id.indexOf(fi_globalid) >= 0) {
					funds_id_map.put(fi_globalid, fi_code);
				}
			}
			rs.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}

		ArrayList<String> dlist = new ArrayList<String>();
		Calendar c = Calendar.getInstance();
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
		Date d = new Date();
		c.setTime(d);
		for (int i = 0; i < 1000; i++) {
			int day = c.get(Calendar.DATE);
			c.set(Calendar.DATE, day - 1);
			//System.out.println(format.format(c.getTime()));
			dlist.add(format.format(c.getTime()));
		}

		for (int i = 0; i < types.size(); i++) {

			PrintStream ps = new PrintStream("./data/tmp/" + String.valueOf(types.get(i)));
			
			StringBuilder sb = new StringBuilder();
			sb.append("fund_code,");
			for (int j = 0; j < dlist.size(); j++) {
				sb.append(dlist.get(j) + ",");
			}
			ps.println(sb.toString());

			ArrayList<Integer> funds = type_funds_id.get(types.get(i));
			for (int j = 0; j < funds.size(); j++) {
				int funds_id = funds.get(j);
				int fi_code = funds_id_map.get(funds_id);
				sql = "select fv_time, fv_total_value from fund_value where fv_fund_id=" + String.valueOf(funds_id);
				HashMap<String, Float> dv = new HashMap<String, Float>();
				try {
					ResultSet rs = reader.selectDB(sql);
					while (rs.next()) {
						String date = rs.getString(1);
						float value = rs.getFloat(2);
						dv.put(date, value);
						// System.out.print(String.valueOf(fi_code) + "\t");
						// System.out.print(String.valueOf(value) + "\t");
					}
					rs.close();
				} catch (SQLException e) {
					e.printStackTrace();
				}
				
				sb = new StringBuilder();
				sb.append(String.format("%06d", fi_code) + ",");
				
				for (int m = 0; m < dlist.size(); m++) {
					String dd = dlist.get(m);
					Float v = dv.get(dd);
					if (v == null) {
						sb.append("0.0000,");
					} else {
						sb.append(String.valueOf(v) + ",");
					}
				}
				ps.println(sb.toString());
				System.out.println(sb.toString());
			}
			ps.close();
		}

		reader.close();
		//st.disconnect();
	}

}

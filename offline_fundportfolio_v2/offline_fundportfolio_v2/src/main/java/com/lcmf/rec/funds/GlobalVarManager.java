package com.lcmf.rec.funds;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import com.lcmf.rec.io.db.RecommendMySQL;

//管理全局变量，全局单例
public class GlobalVarManager {

	private static Logger logger = Logger.getLogger(GlobalVarManager.class);

	private static GlobalVarManager manager = null;

	private List<String> dates_str_list = new ArrayList<String>();

	public GlobalVarManager() {
		// 载入数据
		loaddates();
	}

	public static GlobalVarManager getInstance() {
		if (manager == null) {
			manager = new GlobalVarManager();
			return manager;
		} else {
			return manager;
		}
	}

	
	public void loaddates() {
		
		RecommendMySQL mysql = new RecommendMySQL();

		mysql.connect(RecommendMySQL.host, RecommendMySQL.port, RecommendMySQL.database, RecommendMySQL.username,
				RecommendMySQL.password);
		
		try {
			ResultSet rs = mysql.selectDB("select pv_date from portfolio_values group by pv_date order by pv_date asc");
			while(rs.next()){
				dates_str_list.add(rs.getString(1));
				//System.out.println(rs.getString(1));
			}
		} catch (SQLException e) {
			e.printStackTrace();
		}
		
		mysql.close();
	}

	public List<String> getDates_str_list() {
		return dates_str_list;
	}
	
	

}

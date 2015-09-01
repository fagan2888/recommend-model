package com.lcmf.rec.funds;

import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import com.lcmf.rec.funds.utils.DateStrList;
import com.lcmf.rec.io.db.FundValueReader;

//管理全局变量，全局单例
public class GlobalVarManager {

	private List<String> fund_mofang_ids = new ArrayList<String>(); /** funds mofang id */
	
	private HashMap<String, List<String>> model_fund_value_seq = new HashMap<String, List<String>>(); /** 模型的基金净值表*/

	private List<List<String>> model_fund_values = new ArrayList<List<String>>(); /** 模型的基金净值*/

	private HashMap<String, List<String>> performance_fund_value_seq = new HashMap<String, List<String>>();

	private List<List<String>> performance_fund_values = new ArrayList<List<String>>();
	
	private List<String> performance_date_list = new ArrayList<String>();
	
	private static GlobalVarManager manager = null;
	
	private GlobalVarManager(){
		//载入数据
		loadFundValues();
		performance_date_list = DateStrList.dList(ConstVarManager.getPerformance_start_date_str(), ConstVarManager.getPerformance_end_date_str());
	}
	
	public static GlobalVarManager getInstance(){
		if(manager == null){
			manager = new GlobalVarManager();
			return manager;
		}
		else{
			return manager;
		}
	}

	private void loadFundValues(){
		
		//载入基金净值数据
			try {
				
				//读入model基金净值数据
				FundValueReader reader = new FundValueReader();
				reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,FundValueReader.password);
				reader.readFundIds("./conf/funds");
				
				reader.readFundValues(ConstVarManager.getModel_start_date_str(), ConstVarManager.getModel_end_date_str());
				model_fund_value_seq = reader.getFund_value_seq();
				for (String key : model_fund_value_seq.keySet()) {
					fund_mofang_ids.add(key);
				}
				Collections.sort(fund_mofang_ids);
				for (int i = 0; i < fund_mofang_ids.size(); i++) {
					model_fund_values.add(model_fund_value_seq.get(fund_mofang_ids.get(i)));
				}
				
				//读入performance基金净值数据
				reader.readFundValues(ConstVarManager.getPerformance_start_date_str(), ConstVarManager.getPerformance_end_date_str());
				performance_fund_value_seq = reader.getFund_value_seq();
				for (int i = 0; i < fund_mofang_ids.size(); i++) {
					performance_fund_values.add(performance_fund_value_seq.get(fund_mofang_ids.get(i)));
				}
				
			} catch (SQLException e) {
				e.printStackTrace();
			} catch (ParseException e) {
				e.printStackTrace();
			}
	}

	
	
	public List<String> getPerformance_date_list() {
		return performance_date_list;
	}


	public List<String> getFund_mofang_ids() {
		return fund_mofang_ids;
	}

	public HashMap<String, List<String>> getModel_fund_value_seq() {
		return model_fund_value_seq;
	}

	public List<List<String>> getModel_fund_values() {
		return model_fund_values;
	}

	public HashMap<String, List<String>> getPerformance_fund_value_seq() {
		return performance_fund_value_seq;
	}

	public List<List<String>> getPerformance_fund_values() {
		return performance_fund_values;
	}
	
}

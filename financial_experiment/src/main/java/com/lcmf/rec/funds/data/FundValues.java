package com.lcmf.rec.funds.data;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import com.lcmf.rec.funds.utils.DateStrList;
import com.lcmf.rec.io.db.FundInfoReaer;
import com.lcmf.rec.io.db.FundValueReader;

public class FundValues {

	public static void main(String[] args) throws SQLException, ParseException, FileNotFoundException {
		// TODO Auto-generated method stub

		List<String> datas = new ArrayList<String>();
	
		String start_date_str = "2005-01-01";
		String end_date_str   = "2015-11-05";
		
		FundValueReader v_reader = new FundValueReader();
		v_reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		v_reader.readFundIds("./conf/funds");
		v_reader.readFundValues(start_date_str, end_date_str);
		v_reader.close();

		HashMap<String, List<String>> map = v_reader.getFund_value_seq();

		FundInfoReaer fi_reader = new FundInfoReaer();
		fi_reader.connect(FundInfoReaer.host, FundInfoReaer.port, FundInfoReaer.database, FundInfoReaer.username,
				FundInfoReaer.password);

		for (String key : map.keySet()) {
			List<String> values = map.get(key);
			Integer code = fi_reader.fundCode(Integer.parseInt(key));
			StringBuilder sb = new StringBuilder();
			sb.append(code).append(",");
			for (String value : values) {
				sb.append(value).append(",");
			}

			datas.add(sb.toString());
		}
		
		
		PrintStream ps = new PrintStream("./data/app.csv");

		List<String> dates = DateStrList.dList(start_date_str, end_date_str);

		StringBuilder sb = new StringBuilder();
		sb.append(",");
		for (String date : dates) {
			sb.append(date).append(",");
		}
		ps.println(sb.toString());
		System.out.println(sb.toString());

		for (String str : datas) {
			ps.println(str);
			System.out.println(str);
		}

		ps.flush();
		ps.close();
		
	}

}

package com.lcmf.rec.funds.data;

import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

import com.lcmf.rec.io.db.FundValueReader;

public class Values {

	public static void main(String[] args) throws SQLException, ParseException {
		// TODO Auto-generated method stub

		List<String> datas = new ArrayList<String>();
		
		String start_date_str = "2012-01-04";
		String end_date_str   = "2015-11-03";
		
		FundValueReader v_reader = new FundValueReader();
		v_reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		v_reader.readFundIds("./conf/funds");
		v_reader.readFundValues(start_date_str, end_date_str);
		v_reader.close();
		
	}

}

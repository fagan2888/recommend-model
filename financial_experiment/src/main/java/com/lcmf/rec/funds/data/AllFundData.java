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
import com.lcmf.rec.io.db.FundTypeReader;
import com.lcmf.rec.io.db.FundValueReader;

public class AllFundData {

	private static final int huobi_type = 16;

	private static final int gupiao_type = 5;

	private static final int zhaiquan_type = 7;

	private static final int hunhe_type = 6;

	private static final int etf_type = 9;

	private static final int lof_type = 10;

	private static final int zhishu_type = 15;

	public List<String> typeFund(String start_date_str, String end_date_str, int type)
			throws SQLException, ParseException {

		List<String> datas = new ArrayList<String>();

		FundTypeReader reader = new FundTypeReader();
		reader.connect(FundTypeReader.host, FundTypeReader.port, FundTypeReader.database, FundTypeReader.username,
				FundTypeReader.password);

		List<Integer> funds = reader.typeToFund(type);
		reader.close();

		FundValueReader v_reader = new FundValueReader();
		v_reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		for (Integer id : funds) {
			v_reader.addFundId(String.valueOf(id));
		}
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

		return datas;
	}

	public static void main(String[] args) throws SQLException, ParseException, FileNotFoundException {

		AllFundData afd = new AllFundData();
		List<String> datas = afd.typeFund("2005-01-01", "2015-10-20", AllFundData.huobi_type);
		for (String str : datas) {
			System.out.println(str);
		}

		PrintStream ps = new PrintStream("./data/huobi.csv");

		List<String> dates = DateStrList.dList("2005-01-01", "2015-10-20");

		StringBuilder sb = new StringBuilder();
		sb.append(",");
		for (String date : dates) {
			sb.append(date).append(",");
		}
		ps.println(sb.toString());

		for (String str : datas) {
			ps.println(str);
		}

		ps.flush();
		ps.close();
	}
}
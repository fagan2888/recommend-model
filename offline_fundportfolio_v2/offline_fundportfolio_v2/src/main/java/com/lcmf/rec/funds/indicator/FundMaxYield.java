package com.lcmf.rec.funds.indicator;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;

import com.lcmf.rec.io.db.MoFangMySQLReader;


/**
 * compute funds max yield
 * 
 * @author yjiaoneal
 *
 */

public class FundMaxYield {

	private HashMap<String, ArrayList<Double>> fundValueMap = new HashMap<String, ArrayList<Double>>();

	private HashMap<String, Double> fundMaxYieldMap = new HashMap<String, Double>();

	/**
	 * read all funds history values from licaimofang mysql database
	 */
	public void readAllFundValueFromMysql() {

		MoFangMySQLReader reader = new MoFangMySQLReader();

		reader.connect(MoFangMySQLReader.host, MoFangMySQLReader.port, MoFangMySQLReader.database,
				MoFangMySQLReader.username, MoFangMySQLReader.password);

		String sql = "select fv_fund_id, fv_total_value, fv_time, fv_authority_value from fund_value";

		try {

			HashMap<String, HashMap<Date, Double>> tmp_fdv_map = new HashMap<String, HashMap<Date, Double>>();

			ResultSet rs = reader.selectDB(sql);

			while (rs.next()) {

				String fv_fund_id = rs.getString("fv_fund_id");

				// double fv_tota_value = rs.getDouble("fv_total_value");

				Date fv_time = rs.getDate("fv_time");

				double fv_authority_value = rs.getDouble("fv_authority_value");

				// if authority_value==0 then ignore
				if (fv_authority_value == 0)
					continue;

				HashMap<Date, Double> tmp_dv_map = tmp_fdv_map.get(fv_fund_id);

				if (null == tmp_dv_map) {
					tmp_dv_map = new HashMap<Date, Double>();
				}

				tmp_dv_map.put(fv_time, fv_authority_value);

				tmp_fdv_map.put(fv_fund_id, tmp_dv_map);

			}

			// sort fund value by date
			for (String fv_fund_id : tmp_fdv_map.keySet()) {

				HashMap<Date, Double> tmp_dv_map = tmp_fdv_map.get(fv_fund_id);

				ArrayList<Date> dlist = new ArrayList<Date>(tmp_dv_map.keySet());

				Collections.sort(dlist);

				ArrayList<Double> values = new ArrayList<Double>();

				for (Date d : dlist) {
					values.add(tmp_dv_map.get(d));
				}

				fundValueMap.put(fv_fund_id, values);

			}
		} catch (SQLException e) {
			e.printStackTrace();
		}

		reader.close();

	}

	/**
	 * compute funds values max retrance
	 * 
	 * @param values
	 * @return
	 */

	private static double maxYield(ArrayList<Double> values) {

		double max = 0.0; // max value
		double min = 0.0; // min value
		double retrance = 0.0; // retrance percent;

		for (double v : values) {

			if (v > max) {
				max = v;
				min = v;
			}

			if (v < min) {
				min = v;
			}

			double tmpRetrance = (max - min) / max;

			if (tmpRetrance > retrance) {
				retrance = tmpRetrance;
			}

		}

		return retrance;
	}
	
	public static double maxYield(double[] values) {

		double max = 0.0; // max value
		double min = 0.0; // min value
		double retrance = 0.0; // retrance percent;

		for (double v : values) {

			if (v > max) {
				max = v;
				min = v;
			}

			if (v < min) {
				min = v;
			}

			double tmpRetrance = (max - min) / max;

			if (tmpRetrance > retrance) {
				retrance = tmpRetrance;
			}

		}

		return retrance;
	}

	/**
	 * all funds max yield
	 */
	public void fundMaxYield() {

		for (String fund_id : fundValueMap.keySet()) {

			ArrayList<Double> values = fundValueMap.get(fund_id);

			double maxyield = maxYield(values);

			fundMaxYieldMap.put(fund_id, maxyield);

		}
	}

	public static HashMap<String, Double> allFundMaxYield() {

		FundMaxYield fundyield = new FundMaxYield();

		fundyield.readAllFundValueFromMysql();

		fundyield.fundMaxYield();

		return fundyield.fundMaxYieldMap;
	}

	public static void main(String[] args) throws FileNotFoundException {

		HashMap<String, Double> map = FundMaxYield.allFundMaxYield();

		PrintStream ps = new PrintStream("./data/tmp/fund_yield");

		for (String key : map.keySet()) {
			ps.println(key + "\t" + map.get(key).toString());
		}

		ps.flush();
		ps.close();
		// System.out.println(map);
	}
}

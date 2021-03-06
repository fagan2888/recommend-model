package com.lcmf.rec.funds.markowitz;

import static org.junit.Assert.*;

import java.sql.SQLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.portfolio.FundPortfolio;
import com.lcmf.rec.io.db.FundValueReader;

public class TestMarkowitz {

	@Test
	public void testComputeTime() throws SQLException, ParseException {
		
		long read_start = System.currentTimeMillis();
		FundValueReader reader = new FundValueReader();
		reader.connect(FundValueReader.host, FundValueReader.port, FundValueReader.database, FundValueReader.username,
				FundValueReader.password);
		reader.readFundIds("./conf/funds");
		reader.readFundValues("2006-01-04", "2015-05-30");
		List<List<String>> ds = reader.getValueList();
		long read_end = System.currentTimeMillis();
		System.out.println("Read Data time cost : " + String.valueOf(read_end - read_start));

		int iteration_num = 1000;
		int time = 0;
		for (int i = 0; i < iteration_num; i++) {
			long start = System.currentTimeMillis();
			FundsIndicator fi = new FundsIndicator(ds);
//			long mid = System.currentTimeMillis();
			Markowitz markowitz = new Markowitz(fi.getReturns(), fi.getCov());
			markowitz.setNum(100);
			List<FrontierPoint> fps = markowitz.efficientFrontier();
			long end = System.currentTimeMillis();
			time += end - start;
		}
		System.out.println("compute 1000 markowitz and 5 funds cost time : " + time);
	}
}

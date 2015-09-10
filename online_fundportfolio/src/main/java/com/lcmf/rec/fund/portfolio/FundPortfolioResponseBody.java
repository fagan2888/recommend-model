package com.lcmf.rec.fund.portfolio;

import java.util.ArrayList;
import java.util.List;

public class FundPortfolioResponseBody {
	
	private final long id;
	
	private final String fundportfolio;
	
	private final String content;
	
	private final List<String> list = new ArrayList<String>();
	
	public FundPortfolioResponseBody(long id, String fundportfolio){
		this.id = id;
		this.fundportfolio = fundportfolio;
		this.content = "hehe";
		this.list.add("heihei");
	}

	public long getId() {
		return id;
	}

	public String getFundportfolio() {
		return fundportfolio;
	}

	public String getContent() {
		return content;
	}

	public List<String> getList() {
		return list;
	}
	
}

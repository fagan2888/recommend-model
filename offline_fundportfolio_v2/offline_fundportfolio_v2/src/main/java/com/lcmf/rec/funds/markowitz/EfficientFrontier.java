package com.lcmf.rec.funds.markowitz;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class EfficientFrontier {

	private List<FrontierPoint> list = new ArrayList<FrontierPoint>();
	
	private String ef_name         = "";
	
	
	public List<FrontierPoint> getList() {
		return list;
	}


	public String getEf_name() {
		return ef_name;
	}


	public EfficientFrontier(List<FrontierPoint> list, String type){
		
		this.list = list;
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String today_str = format.format(new Date());
		this.ef_name = "efficientfroniter_" + type + "_" + today_str;
	}
	
	
	public static void main(String[] args) {

	}

}

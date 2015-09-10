package com.lcmf.rec.online_fundportfolio;

import org.apache.log4j.Logger;


//管理全局变量，全局单例
public class GlobalVarManager {

	private static Logger logger = Logger.getLogger(GlobalVarManager.class);
	
	private static GlobalVarManager manager = null;
	
	private GlobalVarManager(){
		//载入数据
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

}

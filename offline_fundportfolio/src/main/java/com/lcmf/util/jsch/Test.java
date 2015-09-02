package com.lcmf.util.jsch;

import java.io.IOException;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

public class Test {

	private static Logger logger = Logger.getLogger(Test.class);
	public static void main(String[] args) throws IOException {

		PropertyConfigurator.configure("./conf/log4j.properties");
		logger.info("Create bench mark portfolio done");
		
		
	}

}

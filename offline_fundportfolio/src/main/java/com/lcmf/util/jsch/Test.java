package com.lcmf.util.jsch;

import java.io.IOException;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

public class Test {

	private static Logger logger = Logger.getLogger(Test.class);

	static {
		PropertyConfigurator.configure("./conf/log4j.properties");
	}

	public static void main(String[] args) throws IOException {

//		while (true) {
//			logger.info("Create bench mark portfolio done");
//		}
		
		
		System.out.println(Math.pow(1.141, 10));
	}

}

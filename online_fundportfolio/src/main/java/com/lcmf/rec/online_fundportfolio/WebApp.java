package com.lcmf.rec.online_fundportfolio;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

public class WebApp {

	private static Logger logger = Logger.getLogger(WebApp.class);

	static {
		PropertyConfigurator.configure("./conf/log4j.properties");
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
	
	}

}

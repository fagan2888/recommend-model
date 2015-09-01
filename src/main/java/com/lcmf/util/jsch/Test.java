package com.lcmf.util.jsch;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

public class Test {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		Properties prop = new Properties();
		FileInputStream fis = new FileInputStream("./conf/mofang.db");
		prop.load(fis);
		System.out.println(prop.getProperty("port","r432432"));
	}

}

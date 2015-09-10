package com.lcmf.rec.funds.io;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;

public class OUT {

	/** 把二维数组输出成csv格式*/
	public static void printCSV(double[][] data, String path) throws FileNotFoundException, UnsupportedEncodingException{
		PrintStream ps = new PrintStream(path, "utf8");
		for(int i = 0; i < data.length; i++){
			StringBuilder sb = new StringBuilder();
			double[] vs = data[i];
			for(int j = 0; j < vs.length; j++){
				sb.append(vs[j]).append(",");
			}
			ps.println(sb.substring(0, sb.length() - 1));
		}
		ps.close();
	}
	
	/** 把二维数组输出*/
	public static void printStdout(double[][] data) throws FileNotFoundException, UnsupportedEncodingException{
		for(int i = 0; i < data.length; i++){
			StringBuilder sb = new StringBuilder();
			double[] vs = data[i];
			for(int j = 0; j < vs.length; j++){
				sb.append(vs[j]).append(",");
			}
			System.out.println(sb.substring(0, sb.length() - 1));
		}
	}
	
	/** 把二维数组输出*/
	public static void printStdout(double[] data) throws FileNotFoundException, UnsupportedEncodingException{
			StringBuilder sb = new StringBuilder();
			for(int j = 0; j < data.length; j++){
				sb.append(data[j]).append(",");
			}
			System.out.println(sb.substring(0, sb.length() - 1));
	}

	
	public static void main(String[] args) {

	}

}
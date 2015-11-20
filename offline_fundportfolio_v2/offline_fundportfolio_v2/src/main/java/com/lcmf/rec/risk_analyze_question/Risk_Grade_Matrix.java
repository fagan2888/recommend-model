package com.lcmf.rec.risk_analyze_question;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import com.lcmf.rec.io.db.RecommendMySQL;

public class Risk_Grade_Matrix {

	
	private List<List<Integer>> list = new ArrayList<List<Integer>>();
	private List<Double> grades = new ArrayList<Double>();
	
	public void readRiskGradeMatrix(String path) throws IOException{
		
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line = null;
		while(null != (line = br.readLine())){
			String[] vec = line.split(",");
			Integer q5 = Integer.parseInt(vec[0]);
			Integer q3 = Integer.parseInt(vec[1]);
			Integer q2 = Integer.parseInt(vec[2]);
			for (int i = 3; i < 7; i++){
				List<Integer> l = new ArrayList<Integer>();
				l.add(q5);
				l.add(q3);
				l.add(q2);
				l.add(i - 2);
				grades.add(Double.parseDouble(vec[i]));
				list.add(l);
			}
		}
		br.close();
	}
	
	public static void main(String[] args) throws IOException, SQLException {
		
		RecommendMySQL mysql = new RecommendMySQL();
		mysql.connect(RecommendMySQL.host, RecommendMySQL.port, RecommendMySQL.database, RecommendMySQL.username,
				RecommendMySQL.password);
		
		String sql_base = "insert into user_risk_grade_matrices (ur_risk_grade, ur_q5, ur_q3, ur_q2, ur_q_ratio, created_at, updated_at) values ('%f', '%d', '%d', '%d', '%d','%s','%s')";
		Risk_Grade_Matrix matrix = new Risk_Grade_Matrix();
		matrix.readRiskGradeMatrix("./data/input/risk_grade_matrix.csv");
		
		Date date = new Date();
		Timestamp tt = new Timestamp(date.getTime());
		
		for(int i = 0; i < matrix.list.size(); i++){
			String sql = String.format(sql_base, matrix.grades.get(i), matrix.list.get(i).get(0), matrix.list.get(i).get(1),matrix.list.get(i).get(2),matrix.list.get(i).get(3),tt.toString(), tt.toString());
			mysql.insertDB(sql);
		}
		
	}

}

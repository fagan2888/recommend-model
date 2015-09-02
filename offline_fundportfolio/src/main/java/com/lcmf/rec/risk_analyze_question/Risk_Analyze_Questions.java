package com.lcmf.rec.risk_analyze_question;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import com.lcmf.rec.io.db.RiskAnalyzeQuestionMySQL;

public class Risk_Analyze_Questions {

	public List<Question> questions = new ArrayList<Question>();

	public void readQuestionsFromFile(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line = null;
		Question q = null;
		while (null != (line = br.readLine())) {
			if (line.indexOf("choice") >= 0) {
				if (q != null) {
					questions.add(q);
				}
				q = new Question();
				String[] vec = line.trim().split("\t");
				q.type = "choice";
				q.question = vec[1].trim();
			} else if (line.indexOf("completion") >= 0) {
				if(q != null){
					questions.add(q);
				}
				q = new Question();
				String[] vec = line.trim().split("\t");
				q.type = "completion";
				q.question = vec[1].trim();
			} else {
				q.options.add(line.trim());
			}
		}
		if(q != null){
			questions.add(q);
		}
		br.close();
	}
	
	public void writeToRecommendMySql(){
		RiskAnalyzeQuestionMySQL writer = new RiskAnalyzeQuestionMySQL();
		writer.connect(RiskAnalyzeQuestionMySQL.host, RiskAnalyzeQuestionMySQL.port, RiskAnalyzeQuestionMySQL.database, RiskAnalyzeQuestionMySQL.username, RiskAnalyzeQuestionMySQL.password);
		for(Question q : questions){
			writer.insertQuestion(q);
		}
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		Risk_Analyze_Questions raq = new Risk_Analyze_Questions();
		raq.readQuestionsFromFile("./data/question/risk_analyze_questions");
		/**
		for(int i = 0; i < raq.questions.size(); i++){
			Question q = raq.questions.get(i);
			System.out.println(q.que stion);
			System.out.println(q.type);
		}
		*/
		raq.writeToRecommendMySql();
	}

}

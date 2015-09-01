package com.lcmf.rec.funds;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import com.lcmf.rec.funds.indicator.FundsIndicator;
import com.lcmf.rec.funds.markowitz.FrontierPoint;
import com.lcmf.rec.funds.markowitz.Markowitz;


/**
 * 按照马科维兹模型计算基金组合有效前沿
 * @author yjiaoneal
 *
 */
public class FundsCombination {

	/**
	 * 原生的有效前沿曲线
	 */
	private List<FrontierPoint> original_efficient_frontier = null;

	/**
	 * 组合个数
	 */
	private int combination_num = 101;
	
	/**
	 * 有效前沿图的点个数
	 */
	private int frontier_num = 50;
	
	
	/**
	 * 基金组合的净值
	 * key : 基金在魔方数据库中id
	 * value : 基金的净值
	 */
	//private HashMap<String, List<String>> fund_value_map = new HashMap<String, List<String>>();
	

	/**
	 * 基金组合净值，按照基金id排序
	 */
	private List<List<String>> fund_values = new ArrayList<List<String>>();

	/**
	 * 时间序列字符串
	 */
	//private List<String> date_list = new ArrayList<String>();
	
	
	/**
	 * 基金代码排序后的结果
	 */
	private List<String> fund_ids = new ArrayList<String>();
	
	/**
	 * 净值开始和结束时间
	 */
//	private String date_start_str = "";
//	private String date_end_str = "";
	

	/**
	 * 
	 * @param fund_values
	 * @param fund_ids
	 */
	public FundsCombination(List<List<String>> fund_values, List<String> fund_ids){
		this.fund_values = fund_values;
		this.fund_ids = fund_ids;
	}

	
	/**
	public static void main(String[] args) throws IOException {
		FundsCombination fc = new FundsCombination();
		List<List<String>> fund_values = fc.readCSVFundsData("./data/input/fund_values.csv");
//		FundsCombination.efficientFrontier(fund_values);
		double max_sd = 0.021;
		double[] sds = new double[100];
		for(int i = 0; i < 100; i++){
			sds[i] = max_sd / 100 * i;
		}
		List<FrontierPoint> fps = FundsCombination.combinations(fund_values, 0.0425 / 365, sds);
		PrintStream ps = new PrintStream(new File("./data/tmp/markowitz.csv"));
		for (FrontierPoint fp : fps) {
			ps.println(fp);
		}
		ps.close();

	}
	**/
	
//	public List<List<String>> readCSVFundsData(String filePath) throws IOException {
//
//		List<List<String>> fund_values = new ArrayList<List<String>>();
//		BufferedReader reader = new BufferedReader(new FileReader(filePath));
//
//		String line = null;
//		while (null != (line = reader.readLine())) {
//			String[] vec = line.split(",");
//			List<String> list = new ArrayList<String>();
//			for (String str : vec) {
//				list.add(str);
//			}
//			fund_values.add(list);
//		}
//		return fund_values;
//	}
	

	/**
	 * 计算有效前沿
	 * @return
	 * @throws FileNotFoundException
	 */
	public List<FrontierPoint> efficientFrontier() {
		
		FundsIndicator fi = new FundsIndicator(this.fund_values);

		Markowitz markowitz = new Markowitz(fi.getReturns(), fi.getCov());
		List<FrontierPoint> fps = markowitz.efficientFrontier();

		original_efficient_frontier = fps;
		
//		try {
//			PrintStream ps = new PrintStream("./data/tmp/frontier_points.csv");
//			for(FrontierPoint fp : fps){
//				StringBuilder sb = new StringBuilder();
//				for(double v : fp.getWeights()){
//					sb.append(v).append(",");
//				}
//				sb.append(",");
//				sb.append(fp.getCamp_sd()).append(",").append(fp.getCamp_return());
//				ps.println(sb.toString());
//			}
//			ps.flush();
//			ps.close();
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
		
		return fps;
	}

	/**
	public static List<FrontierPoint> combinations(List<List<String>> fund_values, double rf, double[] sds) throws FileNotFoundException {

		FundsIndicator fi = new FundsIndicator(fund_values);
		fi.computeCov();
		fi.computeReturns();

		Markowitz markowitz = new Markowitz(fi.getReturns(), fi.getCov());
		List<FrontierPoint> fps = markowitz.efficientFrontier();
		
		PrintStream ps = new PrintStream("./data/tmp/frontier_points");
		for(FrontierPoint fp : fps){
			ps.println(String.valueOf(fp.camp_sd) + "," + String.valueOf(fp.camp_return));
		}
		ps.flush();
		ps.close();

		int point_num = fps.size();
		FrontierPoint shape_fp = Markowitz.perfectShape(fps, rf);
		double shape_risk = shape_fp.camp_sd;
		double shape_return = shape_fp.camp_return;
		double max_return = Double.MIN_VALUE;
		
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.camp_sd < shape_risk){
				fps.remove(i);
				i--;
			}
		}
		
		double re_interval = (shape_return - rf) / 200;
		double ri_interval = shape_risk / 200;
		
		for(int m = 0; m < 200; m++){
			double risk = ri_interval * m;
			double re   = rf + re_interval * m;
			int len = shape_fp.weights.length;
			double[] weights = new double[len];
			for(int n = 0; n < len; n++){
				weights[n] = shape_fp.weights[n] / 200 * m;
			}
			FrontierPoint point = new FrontierPoint();
			point.camp_return = re;
			point.camp_sd = risk;
			point.weights = weights;
			fps.add(point);
		}

		Collections.sort(fps, new Comparator<FrontierPoint>(){
			public int compare(FrontierPoint o1, FrontierPoint o2) {
				if(o1.camp_sd > o2.camp_sd)
					return 1;
				else if(o1.camp_sd < o2.camp_sd)
					return -1;
				else
					return 0;
			}
		});

		
		List<FrontierPoint> results = new ArrayList<FrontierPoint>();
		double max_sd = fps.get(fps.size() - 1).camp_sd;
		for(int i = 0; i < sds.length; i++){
			if(sds[i] > max_sd){
				FrontierPoint p = fps.get(fps.size() - 1).clone();
				p.risk_grade = 1.0 * i / (sds.length - 1);
				results.add(p);
				continue;
			}
			int m = 0;
//			System.out.println(fps.size());
			while(m < fps.size() - 1){
				FrontierPoint lpoint = fps.get(m);
				FrontierPoint rpoint = fps.get(m + 1);
//				System.out.println(lpoint.camp_sd + "\t" + sds[i] + "\t" + rpoint.camp_sd);
				if(lpoint.camp_sd <= sds[i] && rpoint.camp_sd >= sds[i]){
					if(Math.abs(lpoint.camp_sd - sds[i]) < Math.abs(rpoint.camp_sd - sds[i])){
						lpoint.risk_grade = 1.0 * i / (sds.length - 1);
						results.add(lpoint);
					}else{
						rpoint.risk_grade = 1.0 * i / (sds.length - 1);
						results.add(rpoint);
					}
				}
				m++;
			}
		}
		
		return results;
	}
	 * @throws FileNotFoundException 
	**/
	
	public List<FrontierPoint> testfrontierCurve() throws FileNotFoundException{
		
		if(null == original_efficient_frontier){
			this.efficientFrontier();
		}
		
		List<FrontierPoint> fps = new ArrayList<FrontierPoint>();
		for(FrontierPoint fp : original_efficient_frontier)
				fps.add(fp.clone());
		
		double min_sd = Double.MAX_VALUE;
		FrontierPoint min_sd_point = null;
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.getCamp_sd() < min_sd){
				min_sd = fp.getCamp_sd();
				min_sd_point = fp;
			}
		}
		
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.getCamp_return() < min_sd_point.getCamp_return()){
				fps.remove(i);
				i--;
			}
		}
		
		Collections.sort(fps, new Comparator<FrontierPoint>(){
			public int compare(FrontierPoint o1, FrontierPoint o2) {
				if(o1.getCamp_sd() > o2.getCamp_sd())
					return 1;
				else if(o1.getCamp_sd() < o2.getCamp_sd())
					return -1;
				else
					return 0;
			}
		});
		
		
//		PrintStream ps = new PrintStream("./data/tmp/frontier_points.csv");
//		for(FrontierPoint fp : fps){
//			ps.println(String.valueOf(fp.camp_sd) + "," + String.valueOf(fp.camp_return));
//		}
//		ps.flush();
//		ps.close();
		
		return fps;
		
	}
	
	
	/**
	 * 展示给用户看的有效前沿曲线
	 * @return
	 * @throws FileNotFoundException 
	 */
	public List<FrontierPoint> frontierCurve() throws FileNotFoundException{
		
		if(null == original_efficient_frontier){
			this.efficientFrontier();
		}
		
		List<FrontierPoint> fps = new ArrayList<FrontierPoint>();
		for(FrontierPoint fp : original_efficient_frontier)
				fps.add(fp.clone());
		
		double min_sd = Double.MAX_VALUE;
		FrontierPoint min_sd_point = null;
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.getCamp_sd() < min_sd){
				min_sd = fp.getCamp_sd();
				min_sd_point = fp;
			}
		}
		
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.getCamp_return() < min_sd_point.getCamp_return()){
				fps.remove(i);
				i--;
			}
		}
		
		Collections.sort(fps, new Comparator<FrontierPoint>(){
			public int compare(FrontierPoint o1, FrontierPoint o2) {
				if(o1.getCamp_sd() > o2.getCamp_sd())
					return 1;
				else if(o1.getCamp_sd() < o2.getCamp_sd())
					return -1;
				else
					return 0;
			}
		});
		
		List<FrontierPoint> results = new ArrayList<FrontierPoint>();
		int len = fps.size();
		int interval = len / frontier_num;
		int m = 0;
		while(m < len){
			results.add(fps.get(m).clone());
			m = m + interval;
		}
		if(m > len){
			results.add(fps.get(len - 1).clone());
		}
		
		return results;
		
	}
	
	
	public List<FrontierPoint> lowerShapeLine(){
		
		if(null == original_efficient_frontier){
			this.efficientFrontier();
		}
		
		List<FrontierPoint> fps = new ArrayList<FrontierPoint>();
		
		FrontierPoint shape_fp = Markowitz.perfectShape(original_efficient_frontier, ConstVarManager.getRf()).clone();

		double[] ws = new double[shape_fp.getWeights().length];
		
		FrontierPoint no_risk_fp = new FrontierPoint(ConstVarManager.getRf(), 0.0, ws, 0.0);
		
		fps.add(no_risk_fp);
		fps.add(shape_fp);
		
		return fps;
		
	}
	
	/**
	 * 计算用户的组合
	 * @return
	 * @throws FileNotFoundException
	 */
	public List<FrontierPoint> combinations() {

		if(null == original_efficient_frontier){
			this.efficientFrontier();
		}
		
		List<FrontierPoint> fps = new ArrayList<FrontierPoint>();
		for(FrontierPoint fp : original_efficient_frontier)
				fps.add(fp.clone());

		FrontierPoint shape_fp = Markowitz.perfectShape(fps, ConstVarManager.getRf());
		double shape_risk = shape_fp.getCamp_sd();
		double shape_return = shape_fp.getCamp_return();

		
		//去除收益小于shape点收益的点
		for(int i = 0; i < fps.size(); i++){
			FrontierPoint fp = fps.get(i);
			if(fp.getCamp_return() < shape_return){
				fps.remove(i);
				i--;
			}
		}
		
		//添加shape点到无风险收益率点的连线
		double re_interval = (shape_return - ConstVarManager.getRf()) / 200;
		double ri_interval = shape_risk / 200;
		for(int m = 0; m < 200; m++){
			double risk = ri_interval * m;
			double re   = ConstVarManager.getRf() + re_interval * m;
			int len = shape_fp.getWeights().length;
			double[] weights = new double[len];
			for(int n = 0; n < len; n++){
				weights[n] = shape_fp.getWeights()[n] / 200 * m;
			}
			FrontierPoint point = new FrontierPoint(re, risk, weights);
			fps.add(point);
		}

		//按照风险值排序
		Collections.sort(fps, new Comparator<FrontierPoint>(){
			public int compare(FrontierPoint o1, FrontierPoint o2) {
				if(o1.getCamp_sd() > o2.getCamp_sd())
					return 1;
				else if(o1.getCamp_sd() < o2.getCamp_sd())
					return -1;
				else
					return 0;
			}
		});

	
		//每一个组合对应的标准差
		List<FrontierPoint> results = new ArrayList<FrontierPoint>();
		int num = combination_num;
		double max_sd = fps.get(fps.size() - 1).getCamp_sd();
		
		double[] sds = new double[num];
		for (int i = 0; i < num; i++) {
			sds[i] = max_sd / (num - 1) * i;
		}
		
		//根据标准差算出组合
		for(int i = 0; i < sds.length; i++){
			double sd = sds[i];
			
			int m = 0;
			while(m < fps.size() - 1){
				
				FrontierPoint lpoint = fps.get(m);
				FrontierPoint rpoint = fps.get(m + 1);
				
				if(lpoint.getCamp_sd() <= sd && sd <= rpoint.getCamp_sd()){
					if(Math.abs(lpoint.getCamp_sd() - sd) < Math.abs(rpoint.getCamp_sd() - sd)){
						lpoint.setRisk_grade(1.0 * i / (sds.length - 1));
						results.add(lpoint);
					}else{
						rpoint.setRisk_grade(1.0 * i / (sds.length - 1));
						results.add(rpoint);
					}
				}
				m++;
			}
		}
		
		
		
		
		return results;
	}
	
}
package com.lcmf.rec.funds.pca;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import com.lcmf.rec.funds.indicator.COV;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;

/**
 * @author yjiaoneal
 *
 */
public class PCA {

	/** 需要降维的数据 , 按列向量计算*/
	private double[][] data = null;
	
	/** 对称协方差矩阵 */
	private double[][] cov_matrix = null;

	/** 特征值 */
	private double[] eigen_value = null;
	
	/** 特征值组成的对角矩阵*/
	private Matrix eigen_diagonal = null;

	/** 特征向量 */
	private Matrix eigen_vector = null;

	/** pca矩阵 */
	private Matrix pca_value_diagonal = null;

	/** pca特征值 */
	private double[] pca_value = null;

	/** pca特征向量 */
	private Matrix pca_vector_matrix = null;

	/** 主成分*/
	private Matrix pc_matrix = null;
	
	/** 主成分比重 */
	private double threshold = 0.95;
	
	public double[] getPca_value() {
		return pca_value;
	}

	private HashMap<Double, double[]> eigen_value_vector_map = new HashMap<Double, double[]>();
	

	public PCA(double[][] data) {
		this.data = data;
		this.cov_matrix = COV.cov(this.data);
		
		decompose(); // 求特征值和特征向量
		fullEigenMap();
		principalComponent(); // 提取转置矩阵
		pc();            	  // 转置
	}
	
	/** 中心化 */
	public double[][] centered(double[][] data){
		
		int num = data[0].length;
		int len = data.length;
		
		double[][] centered_data = new double[len][num];
		double[] centers = new double[num];
		
		for(int i = 0; i < num; i++){
			double sum = 0;
			for(int j = 0; j < len; j++){
				sum = sum + data[j][i];
			}
			centers[i] = sum / len;
		}
		
		for(int i = 0; i < num; i++){
			for(int j = 0; j < len; j++){
				centered_data[j][i] = data[j][i] - centers[i];
			}
		}
		
		return centered_data;
	} 
	
	/** 计算矩阵特征值和特征向量 */
	public void decompose() {
		
		Matrix matrix = new Matrix(this.cov_matrix);
		EigenvalueDecomposition evDecomposition = new EigenvalueDecomposition(matrix);
		eigen_value = evDecomposition.getRealEigenvalues();
		eigen_vector = evDecomposition.getV();
		eigen_diagonal = evDecomposition.getD();
		
	}

	/** 填入特征值和特征向量对应的map */
	public void fullEigenMap(){
		int len = eigen_vector.getRowDimension();
		for(int i = 0; i < eigen_value.length; i++){
			double v = eigen_value[i];
			double[] vector = new double[len];
			for(int j = 0; j < len; j++){
				vector[j] = eigen_vector.get(j, i);
			}
			eigen_value_vector_map.put(v, vector);
		}
	}
	
	
	/** 提取主成成分 */
	public void principalComponent() {

		double sum = 0;
		for (double v : eigen_value) {
			sum = sum + v;
		}
		
		int num = 0;

		// 对特征值排序
		List<Double> eigen_value_list = new ArrayList<Double>(eigen_value_vector_map.keySet());
		Collections.sort(eigen_value_list, new Comparator<Double>(){
			@Override
			public int compare(Double o1, Double o2) {
				if(o1 < o2){
					return 1;
				}else {
					return -1;
				}
			}
		});
		
		double tmp_sum = 0.0;
		
		for (int i = 0; i < eigen_value_list.size(); i++) {
			double v = eigen_value_list.get(i);
			tmp_sum += v;
			num++;
			if (tmp_sum / sum > threshold) {
				break;
			}
		}

		pca_value_diagonal = new Matrix(num, num, 0.0);
		
		int len = eigen_vector.getRowDimension();

		double[][] pca_vector = new double[num][len];
		for(int i = 0; i < num; i++){
			Double v = eigen_value_list.get(i);
			pca_value_diagonal.set(i, i, v);
			double[] vector = eigen_value_vector_map.get(v);
			for(int j = 0; j < vector.length; j++){
				pca_vector[i][j] = vector[j];
			}
		}
		pca_vector_matrix = new Matrix(pca_vector).transpose();
	}

	
	public void pc(){
		Matrix trans_data = new Matrix(this.data).transpose();
		this.pc_matrix = pca_vector_matrix.copy().transpose().times(trans_data);
	}

	public Matrix getPc_matrix() {
		return pc_matrix;
	}
	
	public double[][] getPc_Array(){
		return pc_matrix.getArrayCopy();
	}

	public Matrix getPca_vector_matrix() {
		return pca_vector_matrix;
	}

	public Matrix getEigen_diagonal() {
		return eigen_diagonal;
	}

	public Matrix getEigen_vector() {
		return eigen_vector;
	}

	public Matrix getPca_value_diagonal() {
		return pca_value_diagonal;
	}
	
}
package com.lcmf.rec.funds.indicator;

public class COV {

	/**
	 * compute covariance
	 * @param a
	 * @param b
	 * @return
	 */
	public static final double cov(double[] a, double[] b){
		
		double ea = 0.0;
		for(double v : a){
			ea += v;
		}
		ea = ea / a.length;
		
		double eb = 0.0;
		for(double v : b){
			eb += v;
		}
		eb = eb / b.length;
		
		double eab = 0.0;
		for(int i = 0; i < b.length; i++){
			eab += a[i] * b[i]; 
		}
		eab = eab / b.length;
		
		double covab = eab - ea * eb;
		
		return covab;
	}
	
	
	public static double variance(double[] values){
		
		double ev = 0.0;
		int len = values.length;
		for(int i = 0; i < len; i++){
			ev = ev + values[i];
		}

		ev = ev / len;
		double variance = 0.0;
		for(int i = 0; i < len; i++){
			variance += (values[i] - ev) * (values[i] - ev);
		}
		
		variance = variance / len;
		
		return variance;
	}
	
}

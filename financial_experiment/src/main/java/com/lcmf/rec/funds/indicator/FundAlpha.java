package com.lcmf.rec.funds.indicator;

/**
 * compute fund alpha
 * @author yjiaoneal
 *
 */

public class FundAlpha {

	public static final double alpha(double fundBeta, double ri, double rf, double rm){
		return (ri - rf) - fundBeta * (rf - rm) ;
	}
	
	public static void main(String[] args) {

	}

}

package com.lcmf.rec.io.excel;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;

public class ExcelReaderFundValue {

	private     List<Integer>      fundId        =           new ArrayList<Integer>();
	
	private     List<Date>        dates         =           new ArrayList<Date>();
	
	private     double[][]        values        =           null;
	
	
	public ExcelReaderFundValue(){}
	
	
	public ExcelReaderFundValue(String filePath, int sheetIndex){
		
		loadExcel(filePath, sheetIndex);
	}
	
	/**
	 * load fund values
	 */
	public      void         loadExcel(String filePath, int sheetIndex){
			
		ExcelReader reader    =    new ExcelReader(filePath);
		
		Sheet       sheet     =    reader.read(sheetIndex);
		
		Row         row       =    sheet.getRow(0);
		
		for (Cell cell : row){
			if(cell  !=  null && cell.getCellType() == Cell.CELL_TYPE_NUMERIC){
				dates.add(cell.getDateCellValue());
			}
		}
		
		
		for (Row arow : sheet){
			Cell cell = arow.getCell(0);
			if(cell != null){
					double v = cell.getNumericCellValue();
					fundId.add((int)(v));
			}
		}

		
		values             =            new  double[fundId.size()][dates.size()];
		
		
		int n = 0;
		for (Row arow : sheet){
			if (n == 0){
				n++;
				continue;
			}
			int m = 0;
			for (Cell cell : arow){
				if (m == 0){
					m++;
					continue;
				}
				double v = cell.getNumericCellValue();
				values[n - 1][m - 1] = v;
				m++;
			}
			n++;
		}
	}
	
	public List<Integer> getFundId() {
		return fundId;
	}


	public List<Date> getDates() {
		return dates;
	}


	public double getValue(int i, int j){
		return values[i][j];
	}
	
	
	
	public static void main(String[] args) {
		ExcelReaderFundValue gupiaoValues = new ExcelReaderFundValue();
		gupiaoValues.loadExcel("./data/input/gupiao_2009.xlsx", 0);
	}

}

package com.lcmf.rec.io.excel;

import java.io.File;
import java.io.IOException;
import org.apache.poi.EncryptedDocumentException;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;

public class ExcelReader {

	Workbook wb = null;

	public ExcelReader(String filePath) {
		loadExcel(filePath);
	}

	private void loadExcel(String filePath) {
		try {
			wb = WorkbookFactory.create(new File(filePath));
		} catch (EncryptedDocumentException | InvalidFormatException | IOException e) {
			e.printStackTrace();
		}
	}

	public Sheet read(int sheetIndex) {
		Sheet sheet = wb.getSheetAt(0);
		return sheet;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		ExcelReader reader = new ExcelReader("./data/input/gupiao_2009.xlsx");
		reader.read(0);
	}

}

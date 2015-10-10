package com.lcmf.rec.online_fundportfolio;

import java.io.FileNotFoundException;
import java.io.IOException;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@ComponentScan({"com.lcmf.rec.web_controller","com.lcmf.rec.schedule_task"})
@EnableScheduling
public class WebApp {

	private static Logger logger = Logger.getLogger(WebApp.class);

	static {
		PropertyConfigurator.configure("./conf/log4j.properties");
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		logger.info("web app starting");
		ApplicationContext context = SpringApplication.run(WebApp.class, args);
		logger.info("web app start done");
	}


	
}

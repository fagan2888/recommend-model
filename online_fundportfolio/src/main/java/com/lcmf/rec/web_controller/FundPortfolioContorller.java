package com.lcmf.rec.web_controller;

import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import com.lcmf.rec.fund.portfolio.FundPortfolioResponseBody;

@Controller
@RequestMapping("/fund_portfolio")
public class FundPortfolioContorller {

	private final AtomicLong counter = new AtomicLong();
	
	@RequestMapping(method=RequestMethod.GET)
	public @ResponseBody FundPortfolioResponseBody fundPortfolio(@RequestParam(value="fundportfolio", defaultValue="Fund Portfolio") String fundportfolio_str){
		return new FundPortfolioResponseBody(counter.getAndIncrement(), fundportfolio_str);
	}
	
}

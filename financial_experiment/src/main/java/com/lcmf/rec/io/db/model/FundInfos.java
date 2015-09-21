package com.lcmf.rec.io.db.model;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
@Table(name = "fund_infos")
public class FundInfos {

	
    @Id
    @GeneratedValue(strategy=GenerationType.IDENTITY)
    @Column(name="id", nullable = false)
    private int id;
    
	@Column(name="fi_globalid")
	private long fi_globalid;
	
	@Column(name="fi_code")
	private int fi_code;
	
	@Column(name="fi_name")
	private String fi_name;

	public FundInfos(long fi_globalid, int fi_code, String fi_name){
		this.fi_globalid = fi_globalid;
		this.fi_code = fi_code;
		this.fi_name = fi_name;
	}
	
	public FundInfos(){};
	
	public long getFi_globalid() {
		return fi_globalid;
	}

	public void setFi_globalid(long fi_globalid) {
		this.fi_globalid = fi_globalid;
	}

	public int getFi_code() {
		return fi_code;
	}

	public void setFi_code(int fi_code) {
		this.fi_code = fi_code;
	}

	public String getFi_name() {
		return fi_name;
	}

	public void setFi_name(String fi_name) {
		this.fi_name = fi_name;
	}
	
}
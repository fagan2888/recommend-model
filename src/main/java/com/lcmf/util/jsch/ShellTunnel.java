/* -*-mode:java; c-basic-offset:2; indent-tabs-mode:nil -*- */
/**
 * This program enables you to connect to sshd server and get the shell prompt.
 *   $ CLASSPATH=.:../build javac Shell.java 
 *   $ CLASSPATH=.:../build java Shell
 * You will be asked username, hostname and passwd. 
 * If everything works fine, you will get the shell prompt. Output may
 * be ugly because of lacks of terminal-emulation, but you can issue commands.
 *
 */

package com.lcmf.util.jsch;


import com.jcraft.jsch.*;


public class ShellTunnel{
	
	private JSch jsch = new JSch();
	
	private Session session = null;
	
	public boolean connectSSH(String host, String user, String password, int port){
		
		try {
			
			session = jsch.getSession(user, host, port);
			session.setPassword(password);
			session.setConfig("StrictHostKeyChecking", "no");
			session.connect(30000);
			return true;
			
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (JSchException e){
			e.printStackTrace();
		}

		return false;
		
	}
	
	public void disconnect(){
		session.disconnect();
	}
	
	public boolean portForwarding(int lport, String host, int rport){
		try {
			session.setPortForwardingL(lport, host, rport);
			return true;
		} catch (JSchException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	public static void main(String[] args){
		ShellTunnel st = new ShellTunnel();
		st.connectSSH("182.92.214.1", "jiaoyang", "jiao123456", 22);
		System.out.println(st.session.getServerVersion());
	}
}

package com.lcmf.rec.fundpool;


import java.util.List;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;

import com.lcmf.rec.io.db.model.FundInfos;
import junit.framework.TestCase;

public class TestFundInfos extends TestCase{

	private SessionFactory sessionFactory;
	
	@Override
	protected void setUp() throws Exception {
//			final StandardServiceRegistry registry = new StandardServiceRegistryBuilder()
//					.configure() // configures settings from hibernate.cfg.xml
//					.build();
//			try {
//				MetadataSources ms = new MetadataSources(registry);
//				System.out.println(ms.getAnnotatedClassNames());
//				sessionFactory = new MetadataSources( registry ).buildMetadata().buildSessionFactory();
//			}
//			catch (Exception e) {
//				// The registry would be destroyed by the SessionFactory, but we had trouble building the SessionFactory
//				// so destroy it manually.
//				StandardServiceRegistryBuilder.destroy( registry );
//			}
		
		sessionFactory = new Configuration().configure().buildSessionFactory();
	}

	@Override
	protected void tearDown() throws Exception {
		if ( sessionFactory != null ) {
			sessionFactory.close();
		}
	}

	@SuppressWarnings({ "unchecked" })
	public void testBasicUsage() {

		Session session = sessionFactory.openSession();
        session.beginTransaction();
        session.save(new FundInfos(1,2,"fund_AAA"));
        session.getTransaction().commit();
        session.close();
        
		session = sessionFactory.openSession();
        session.beginTransaction();
        List result = session.createQuery( "from FundInfos" ).list();
		for ( FundInfos info : (List<FundInfos>) result ) {
			System.out.println( "FundInfos (" + info.getFi_name() + ")");
		}
        session.getTransaction().commit();
        session.close();
	}
}

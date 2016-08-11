#coding=utf8
import sys
sys.path.append('shell')
from collections import OrderedDict
import DB
import copy
import time
import datetime

def cost(all_code_position):
    risk_day = {}
    risk_position = {}
    risk_position_other = {}
    risk_position_buy_po_fee = {}
    for record in all_code_position:
        if risk_day.has_key(str(record[0])):
            if str(record[1]) not in risk_day[str(record[0])]:
                risk_day[str(record[0])].append(str(record[1]))
        else:
            risk_day[str(record[0])] = [str(record[1]),]
        if risk_position.has_key(str(record[0])+'--'+str(record[1])):
            risk_position[str(record[0])+'--'+str(record[1])][str(record[2])] = record[3]
            risk_position_other[str(record[0])+'--'+str(record[1])] += record[3]
        else:
            risk_position[str(record[0])+'--'+str(record[1])] = {}
            risk_position[str(record[0])+'--'+str(record[1])][str(record[2])] = record[3]
            risk_position_other[str(record[0])+'--'+str(record[1])] = record[3]
        if float(record[3])>=0.1:
            po_fee = DB.getBuyPoFee(str(record[2]))/float(record[3])
        else:
            po_fee = 0
        if risk_position_buy_po_fee.has_key(str(record[0])+'--'+str(record[1])):
            if risk_position_buy_po_fee[str(record[0])+'--'+str(record[1])] < po_fee:
                risk_position_buy_po_fee[str(record[0])+'--'+str(record[1])] = po_fee
        else:
            risk_position_buy_po_fee[str(record[0])+'--'+str(record[1])] = po_fee
   
    #po_dict = sorted(risk_position_buy_po_fee.iteritems(), key=lambda d:d[1], reverse = True)
    #for i in po_dict:
    #    print i 

    for k,v in risk_position_other.items():
        if v<1:
            risk_position[k][str(213009)] = 1-v    

    risk_position_cost = {}
    for k,v in risk_day.items():
        #if k != str(0.1) and k != str(0.2):
        #    continue
        v.sort() 
        laste = {}
        code_list = []
        share_list = []
        time_list = []
        all_fee = 0
        all_cost = 0
        last_amount = 0
        change_times = 0
        fee_dict = {}
        for i in v:
            #if isDay('2013-08-22',i)<0:
            if isDay('2013-07-01',i)<=0:
                continue 
            #print k,i,'配置',risk_position[k+'--'+i]
            change_times +=1
            if laste == {}:
            #    print i
                laste = risk_position[k+'--'+i]
                risk_position_cost[k+'--'+i] = 0 
                fee = 0
                for j,l in risk_position[k+'--'+i].items():
                    tmp_fee = DB.getFee(j,0,10000*l)
                    fee += tmp_fee 
                    fund_type = DB.getFundType(j)
                    if fee_dict.has_key(fund_type):
                        fee_dict[fund_type]['buy'] += tmp_fee
                    else:
                        fee_dict[fund_type] = {'buy':0,'del':0}
                        fee_dict[fund_type]['buy'] += tmp_fee
                    code_list.append(j)
                    [day,share] = DB.getShare(j,10000*l-tmp_fee,i)
                    share_list.append(share)
                    time_list.append(day)
                risk_position_cost[k+'--'+i] = fee/(10000)
                all_fee = fee
                all_cost = fee/(10000)
                #print '成本:',10000,'调仓花费:',fee,'花费占比:',fee/10000
            else: 
            #    continue
                cg = change(code_list,share_list,time_list,risk_position[k+'--'+i],i)
                #print '成本:',cg['amount'],'调仓花费:',cg['fee'],'花费占比:',cg['cost']
            #    if cg['fee_dict'].has_key('zhaiquan') and cg['fee_dict']['zhaiquan']['del']/cg['amount'] >=0.02:
            #        continue
                code_list = cg['code_list']
                share_list = cg['share_list']
                time_list = cg['time_list']
                tmp = getUpdateShare(code_list,share_list,time_list,cg['dict'],i,cg['amount']-cg['fee'],risk_position[k+'--'+i])
                code_list = tmp['code_list']
                share_list = tmp['share_list']
                time_list = tmp['time_list']
                all_fee += cg['fee']
                all_cost += cg['cost']
                for m,n in cg['fee_dict'].items():
                    if fee_dict.has_key(m):
                        fee_dict[m]['buy'] += n['buy']
                        fee_dict[m]['del'] += n['del']
                    else:
                        fee_dict[m] = n
        for code,share,time in zip(code_list,share_list,time_list):
            amount = DB.getAmount(code,share)
            last_amount += amount
        print k,all_fee,last_amount#,all_cost,change_times,fee_dict


def change(code_list_origin,share_list_origin,time_list_origin,position_origin,day):
    code_list = code_list_origin[:]
    share_list = share_list_origin[:]
    time_list = time_list_origin[:]
    position = position_origin
    company_list = {}
    amount_list = []
    amounts = 0
    old_key = {}
    fee = 0
    change_dict = []
    fee_dict = {}
    for code,share,time in zip(code_list,share_list,time_list):
        amount = DB.getAmount(code,share,day)
        amounts += amount
    index = 0
    index_list = []
    del_index_list = {}
    for code,share,time in zip(code_list,share_list,time_list):
        company_id = DB.getCompany(code)
        amount = DB.getAmount(code,share,day)
        amount_list.append(amount)
        if old_key.has_key(code):
            old_key[code] += amount
        else:
            old_key[code] = amount
        if position.has_key(code):
            pass
        else:
            tmp_fee = DB.getFee(code,1,amount,getDelDay(time,day))
            fee += tmp_fee
            fund_type = DB.getFundType(code)
            if fee_dict.has_key(fund_type):
                fee_dict[fund_type]['del'] += tmp_fee
            else:
                fee_dict[fund_type] = {'buy':0,'del':0}
                fee_dict[fund_type]['del'] += tmp_fee
            if DB.isChangeOut(code):  
                if company_list.has_key(company_id):
                    company_list[company_id] += amount
                else:
                    company_list[company_id] = amount
            change_dict.append({'code':code,'amount':amount/amounts,'type':'del'})
            index_list.append(index)
        index +=1
         
    for k,v in position.items():
        change_amount = amounts*v
        if old_key.has_key(k):
            if old_key[k] >= change_amount:  
                del_amount = old_key[k] - change_amount
                index = 0
                for code,share,time in zip(code_list,share_list,time_list):
                    if code == k:
                        company_id = DB.getCompany(code)
                        amount = DB.getAmount(code,share,day)
                        nav_value = DB.getNavValue(code,day)
                        if amount >= del_amount:
                            del_share = del_amount/nav_value
                            tmp_fee = DB.getFee(code,1,del_amount,getDelDay(time,day))
                            fee += tmp_fee
                            fund_type = DB.getFundType(code)
                            if fee_dict.has_key(fund_type):
                                fee_dict[fund_type]['del'] += tmp_fee
                            else:
                                fee_dict[fund_type] = {'buy':0,'del':0}
                                fee_dict[fund_type]['del'] += tmp_fee
                            del_amount = 0
                            if DB.isChangeOut(code):  
                                if company_list.has_key(company_id):
                                    company_list[company_id] += del_amount
                                else:
                                    company_list[company_id] = del_amount
                            change_dict.append({'code':code,'amount':del_amount/amounts,'type':'del'})
                            if del_index_list.has_key(index):
                                del_index_list[index] += del_share 
                            else:
                                del_index_list[index] = del_share 
                        else:
                            tmp_fee = DB.getFee(code,1,amount,getDelDay(time,day))
                            fee += tmp_fee
                            fund_type = DB.getFundType(code)
                            if fee_dict.has_key(fund_type):
                                fee_dict[fund_type]['del'] += tmp_fee
                            else:
                                fee_dict[fund_type] = {'buy':0,'del':0}
                                fee_dict[fund_type]['del'] += tmp_fee
                            del_amount -= amount
                            change_dict.append({'code':code,'amount':amount/amounts,'type':'del'})
                            index_list.append(index)
                    index += 1

    for k,v in del_index_list.items():
        share_list[k] -= v

    index_list.sort(reverse=True)
    for i in index_list:
        del(code_list[i])
        del(share_list[i])
        del(time_list[i])
        

    for k,v in position.items():
        company_id = DB.getCompany(k)
        change_amount = amounts*v
        if old_key.has_key(k):
            if old_key[k] < change_amount:  
                add_amount = change_amount - old_key[k] 
                if DB.isChangeIn(k):
                    if company_list.has_key(company_id):
                        if company_list[company_id] > add_amount:
                            change_dict.append({'code':k,'amount':add_amount/amounts,'type':'add','time':1})
                            company_list[company_id] -= add_amount
                            continue
                        else:
                            tmp_fee = DB.getFee(k,0,add_amount-company_list[company_id])
                            fee += tmp_fee
                            fund_type = DB.getFundType(k)
                            if fee_dict.has_key(fund_type):
                                fee_dict[fund_type]['buy'] += tmp_fee
                            else:
                                fee_dict[fund_type] = {'buy':0,'del':0}
                                fee_dict[fund_type]['buy'] += tmp_fee
                            change_dict.append({'code':k,'amount':company_list[company_id]/amounts,'type':'add','time':1})
                            change_dict.append({'code':k,'amount':(add_amount-company_list[company_id])/amounts,'type':'add','time':3})
                            del(company_list[company_id])
                    else:
                        tmp_fee = DB.getFee(k,0,add_amount)
                        fee += tmp_fee
                        fund_type = DB.getFundType(k)
                        if fee_dict.has_key(fund_type):
                            fee_dict[fund_type]['buy'] += tmp_fee
                        else:
                            fee_dict[fund_type] = {'buy':0,'del':0}
                            fee_dict[fund_type]['buy'] += tmp_fee
                        change_dict.append({'code':k,'amount':add_amount/amounts,'type':'add','time':3})
                else:
                    tmp_fee = DB.getFee(k,0,add_amount)
                    fee += tmp_fee
                    fund_type = DB.getFundType(k)
                    if fee_dict.has_key(fund_type):
                        fee_dict[fund_type]['buy'] += tmp_fee
                    else:
                        fee_dict[fund_type] = {'buy':0,'del':0}
                        fee_dict[fund_type]['buy'] += tmp_fee
                    change_dict.append({'code':k,'amount':add_amount/amounts,'type':'add','time':3})
        else:
            if DB.isChangeIn(k):
                if company_list.has_key(company_id):
                    if company_list[company_id] > change_amount:
                        change_dict.append({'code':k,'amount':change_amount/amounts,'type':'add','time':1})
                        company_list[company_id] -= change_amount
                        continue
                    else:
                        tmp_fee = DB.getFee(k,0,change_amount-company_list[company_id])
                        fee += tmp_fee
                        fund_type = DB.getFundType(k)
                        if fee_dict.has_key(fund_type):
                            fee_dict[fund_type]['buy'] += tmp_fee
                        else:
                            fee_dict[fund_type] = {'buy':0,'del':0}
                            fee_dict[fund_type]['buy'] += tmp_fee
                        change_dict.append({'code':k,'amount':company_list[company_id]/amounts,'type':'add','time':1})
                        change_dict.append({'code':k,'amount':(change_amount-company_list[company_id])/amounts,'type':'add','time':3})
                        del(company_list[company_id])
                else:
                    tmp_fee = DB.getFee(k,0,change_amount)
                    fee += tmp_fee
                    fund_type = DB.getFundType(k)
                    if fee_dict.has_key(fund_type):
                        fee_dict[fund_type]['buy'] += tmp_fee
                    else:
                        fee_dict[fund_type] = {'buy':0,'del':0}
                        fee_dict[fund_type]['buy'] += tmp_fee
                    change_dict.append({'code':k,'amount':change_amount/amounts,'type':'add','time':3})
            else:
                tmp_fee = DB.getFee(k,0,change_amount)
                fee += tmp_fee
                fund_type = DB.getFundType(k)
                if fee_dict.has_key(fund_type):
                    fee_dict[fund_type]['buy'] += tmp_fee
                else:
                    fee_dict[fund_type] = {'buy':0,'del':0}
                    fee_dict[fund_type]['buy'] += tmp_fee
                change_dict.append({'code':k,'amount':change_amount/amounts,'type':'add','time':3})
    
    return {'fee':fee,'cost':fee/amounts,'dict':change_dict,'amount':amounts,'code_list':code_list,'share_list':share_list,'time_list':time_list,'fee_dict':fee_dict} 
                        
def getUpdateShare(code_list,share_list,time_list,change_dict,day,amount,position):
    for i in change_dict:
        if i['type'] == 'add':
            code_list.append(i['code'])   
            [time,share] = DB.getShare(i['code'],amount*i['amount'],day,i['time'])
            share_list.append(share)   
            time_list.append(time)   
        
    return {'code_list':code_list,'share_list':share_list,'time_list':time_list}
    
def getDelDay(day1,day2):
    date1 = time.strptime(str(day1),"%Y-%m-%d")          
    date2 = time.strptime(str(day2),"%Y-%m-%d %H:%M:%S")          
    d1 = datetime.datetime(date1[0],date1[1],date1[2])
    d2 = datetime.datetime(date2[0],date2[1],date2[2])
    day = (d2 - d1).days
    return abs(day)
       
def isDay(day1,day2):
    date1 = time.strptime(str(day1),"%Y-%m-%d")          
    if len(day2)==10:
        date2 = time.strptime(str(day2),"%Y-%m-%d")          
    else:
        date2 = time.strptime(str(day2),"%Y-%m-%d %H:%M:%S")          
    d1 = datetime.datetime(date1[0],date1[1],date1[2])
    d2 = datetime.datetime(date2[0],date2[1],date2[2])
    day = (d2 - d1).days
    return day

    
if __name__ == '__main__':
 
    print getDelDay('2016-01-03','2016-01-01')
    #getFee(fund_id,fee_type,amount,day=0)


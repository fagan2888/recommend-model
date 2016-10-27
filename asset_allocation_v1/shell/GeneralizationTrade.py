#coding=utf8
import sys
sys.path.append('shell')
import DB
import time

def readFile(path):
    tmp = []
    if type(path) == list: 
        for line in path:   
            tmp.append(line.strip().split(','))
    else:
        for line in open(path):   
            tmp.append(line.strip().split(','))
    return tmp

def loadDict(path):
    tmp = readFile(path)
    result = {}
    day_list = {}
    risk_list = []
    codes = []
    days = []
    for i in tmp:
        l = i[4].strip().split(':')
        codes.extend(l)
        days.append(str(i[0]))
    days = list(set(days))
    codes = list(set(codes))
    buy_list = getNotBuy(days,codes) 
    for i in tmp:
        if float(i[3]) == 0:
            continue
        key = str(i[1])+'--'+str(i[0])
        if not result.has_key(key):  
            l = i[4].strip().split(':')
            l = cleanNotBuy(i[0],l,buy_list) 
            if l == []:
                raise Exception("the funds is can not buy in risk:%s,day:%s,list:%s" % (i[1],i[0],i[4]))
            result[key] = {str(i[2]):{'sum':i[3],'list':l}}
        else:
            l = i[4].strip().split(':')
            l = cleanNotBuy(i[0],l,buy_list) 
            if l == []:
                raise Exception("the funds is can not buy in risk:%s,day:%s,list:%s" % (i[1],i[0],i[4]))
            result[key][str(i[2])] = {'sum':i[3],'list':l}
        if day_list.has_key(str(i[1])):
            if str(i[0]) not in day_list[str(i[1])]:
                day_list[str(i[1])].append(str(i[0]))
        else:
            day_list[str(i[1])] = [str(i[0]),]
        if str(i[1]) not in risk_list:
            risk_list.append(str(i[1]))
    return result,day_list,risk_list

def getNotBuy(days,funds):
    return DB.getBuyStatus(days,funds)

def cleanNotBuy(day,funds,buy_list):
    date1 = time.strptime(str(day),"%Y-%m-%d")
    date2 = time.strptime(str('2016-09-03'),"%Y-%m-%d")
    if date1 < date2:
        # day = '2016-09-03'
        return funds
    tmp = []
    for fund in funds:
        if fund in ['096001', '320013', '000216', '000071', '213009']:
            tmp.append(fund)
        else:
            if buy_list.has_key(day):
                if fund in buy_list[day]:
                    tmp.append(fund)
    return tmp
                
def getTrade(position,day_list,risk_list): 
    result = []
    for risk in risk_list:
        old_position = [] 
        for day in day_list[risk]:
            key = risk+'--'+day
            tmp = []
            for k,v in position[key].items():
                ge = getGeneralization(old_position,v,risk,day)
                tmp.extend(ge)
            codes,tmp = merge(tmp) 
            result.extend(tmp)
            old_position = codes
    return result
                    
                    
            
def getGeneralization(old,position,risk,day):
    tmp = list(set(position['list']).intersection(set(old)))
    ratio = position['sum']
    tmp_list = []
    if tmp:
        tmo_list = tmp
        for i in position['list']:
            if i not in tmp_list:
                tmp_list.append(i) 
    else:
        tmp_list = position['list']

    if ratio > 0.60 :
        count_used = min (5, len(tmp_list))
    elif ratio > 0.45 :
        count_used = min (4, len(tmp_list))
    elif ratio > 0.30 :
        count_used = min (3, len(tmp_list))
    elif ratio > 0.15 :
        count_used = min (2, len(tmp_list))
    else :
        count_used = 1;

    codes_used = tmp_list[0:count_used]
    ratio_used = float(ratio) / count_used
    result = []
    for i in codes_used:
        result.append((risk,day,i,round(ratio_used,4)))
    return result
        
    

def merge(position):
    tmp = {}
    risk = None
    day = None
    for i in position:
        risk = str(i[0])
        day = str(i[1])
        if tmp.has_key(i[2]):
            tmp[i[2]] += i[3]
        else:
            tmp[i[2]] = i[3]
    result = []
    for k,v in tmp.items():
        result.append((risk,day,str(k),round(v,4)))
    return tmp.keys(),result
         
            
         


def init(path):
    position,day_list,risk_list = loadDict(path)
    return getTrade(position,day_list,risk_list)

if __name__ == '__main__':
    print init('tmp/gposition-20160830-0.csv')




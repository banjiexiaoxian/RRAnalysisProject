import pymysql
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
#Connect to the database
connection = pymysql.connect(host='120.77.210.240',
                             port=3306,
                             user='pku',
                             passwd='pkuims',
                             db='pku_heal',
                             charset = 'utf8mb4',
                             cursorclass = pymysql.cursors.DictCursor)

startTime =  datetime(2017, 11,29,13, 20, 00)
endTime = datetime(2017, 11, 29,14, 20, 59)
userName = '刘言灵'

# 执行sql语句
try:
    with connection.cursor() as cursor:
        # 执行sql语句，插入记录
        sql = 'SELECT RR FROM heal_app_additional_data WHERE FuID IN ' \
              '(SELECT ID FROM heal_app_data AS a LEFT JOIN heal_user AS b ON a.YongHuID = b.YongHuID ' \
              'WHERE b.XingMing LIKE %s AND a.TianJianShiJian BETWEEN %s AND %s )'
        cursor.execute(sql, (userName,startTime, endTime))
        # 获取查询结果
        result = cursor.fetchall()
        li = []
        rr = []
        for i in result:
            li.append(i['RR'])
            rr.append(i['RR'])
        rr = np.array(rr)
        rr = rr / 1000
        rr = rr[rr < 2]
        rr = rr[rr > 0.3]
        rr_i = rr[0:len(rr) - 1]
        rr_j = rr[1:len(rr)]
        plt.plot(rr_i,rr_j, 'o', markerfacecolor='k', markeredgecolor='k', markersize=3,
                 )
        liststr = [startTime.strftime('%y_%m_%d_%H_%M'), endTime.strftime('%y_%m_%d_%H_%M')]
        plt.title(liststr)
        plt.xlabel('RR(i)(sec)', fontsize=15)
        plt.ylabel('RR(i+1)(sec)', fontsize=15)
        # plt.xlim(0.3, 2)
        # plt.ylim(0.3, 2)
        plt.show()
        plt.plot(rr,'o', markerfacecolor='k', markeredgecolor='k', markersize=3)
        # plt.ylim(0.3, 2)
        plt.show()
        # print(li)
        # liststr = [userName,startTime.strftime('%y_%m_%d_%H_%M'),endTime.strftime('%y_%m_%d_%H_%M'),'.txt']
        # f = open(''.join(liststr), 'w')
        # for i in li:
        #     f.write(str(i) + "\n")
        # f.close()
        ##print(list(result))
    # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
    connection.commit()

finally:
    connection.close()
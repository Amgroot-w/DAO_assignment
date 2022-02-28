"""
《深度学习》作业 --- ”小球称重问题“
              --- 基于『信息熵』给出所需次数，基于『贪心思想』给出解决方案


调试记录：

"""
import random

def InputBallnum():
    # 输入球数，并计算所需称重次数，返回球数，称重次数，除以三的余数
    n = 12
    print('小球数量：', n)

    """
    n个球，找到异常球的信息熵是：log(n)
    找到异常球后得知轻重信息的信息熵是：log(2)
    所以问题的总信息熵: S1 = log(n) * log(2) = log(2n)
    
    每次称重包含左-右-平衡三种情况 所以每次称重的信息熵是：log(3)
    k次称重的总信息熵: S2 = k * log(3)
    
    要想在k次称重后找出异常球，那么需要保证k次称重的总信息熵S2大于问题的总信息熵S1，即：
    k * log(3) > log(2n)   ===>   k > log(2n/3)
    
    当 n = 12 时，解得：k >= 3
    所以，12个小球的称重问题，称3次即可找到异常球，并能够确定它是过重还是过轻。
    """
    m = n % 3
    for times in range(n):
        if (3 ** times - 3) / 2 >= n:
            print("%d个球%d次可以完成称重！" % (n, times))
            print()
            return times, n, m


def StarCode(t):
    # 输入称重次数即编码位数t，返回所有t位首位为0的正序编码
    if t >= 3:
        star = ['0', '1', '2']  # 初始化列表，元素为三进制的三个字符
        str1 = Make_Num(t - 2, star)  # 产生一个t-1位由0,1,2组成的所有编码组合的列表
        for i in range(len(str1)):
            str1[i] = '0' + str1[i]  # 在t位所有编码前添加一个0
        for string in str1[:]:
            # 从列表中排除所有非正序的编码
            for i in range(t - 1):
                if (int(string[i + 1]) - int(string[i]) == 1) or (int(string[i + 1]) - int(string[i])) == -2:
                    # 后一个数字减前一个数字为1或-2，则该编码为正序
                    break
                elif (int(string[i + 1]) - int(string[i]) == -1) or ((int(string[i + 1]) - int(string[i])) == 2) or (
                        (i == t - 2) and (int(string[i + 1]) - int(string[i])) == 0):
                    # 后一个数字减前一个数字为-1或2，或所有数字相同，则该序列非正序
                    str1.remove(string)
                    break
        return str1
    else:
        return star


def Make_Num(n, list):
    # 利用递归调用生成n+1位由0,1,2组成的所有编码组合的列表
    if n:
        for i in range(len(list)):
            for j in range(3):
                list.append(list[i] + str(j))
        for i in range(int(len(list) / 4)):
            list.remove(list[0])
        Make_Num(n - 1, list)
        return list


def FinalCode(n, list, m):
    # 输入球数n，首位为0的t位正序码列表，余数，返回n个符合条件的正序码
    # 该步骤为将0替换为1，1替换为2,2替换为0，并将新编码加入列表
    if m == 0:
        t = n // 3
    else:
        t = n // 3 + 1
    templist = []
    for i in range(t):
        templist.append(list[i])
        temp1 = temp2 = ''
        for j in range(len(list[0])):
            # 第一层替换
            l = int(list[i][j]) + 1
            if l == 3:
                l = 0
            temp1 += str(l)
        for k in range(len(temp1)):
            # 第二层替换
            ll = int(temp1[k]) + 1
            if ll == 3:
                ll = 0
            temp2 += str(ll)
        templist.append(temp1)
        templist.append(temp2)
    return templist


def SetBall(n, list, m):
    # 生成表示小球的列表并返回，元素分别为编号、质量、编码
    ball = []
    for i in range(n - m):
        fig = '(' + str(i + 1) + ')'  # 小球编号表示如"(10)"
        # 按顺序将以上产生的正序编码赋给每个小球，质量设为10
        ball.append([fig, 10, list[i]])
    if m == 1:
        # 当球数非三的倍数时，需额外生成多余的小球
        ball.append(['(' + str(n - m + 1) + ')', 10, list[-1]])
    elif m == 2:
        ball.append(['(' + str(n - m + 1) + ')', 10, list[-3]])
        ball.append(['(' + str(n - m + 2) + ')', 10, list[-2]])
    return ball


def ChangeWeight(list):
    # 改变一个小球的质量，手动输入或随机生成
    n = len(list)
    select = 'N'
    while select not in 'YyNn':
        select = input("输入有误！请重新输入 >> ")
    if select == 'Y' or select == 'y':
        num = int(input("请在1~%d中选择一个球，改变其质量 >> " % n))
        while num not in range(1, n + 1):
            num = int(input("输入有误！请重新选择 >> "))
        weit = eval(input("请选择增加质量(1)或减少质量(-1) >> "))
        while weit != 1 and weit != -1:
            weit = eval(input("输入有误！请重新输入 >>　"))
        list[num - 1][1] += weit
    elif select == 'N' or select == 'n':
        # 随机取一个小球，将其质量+1或-1
        list[random.randint(0, n - 1)][1] += random.choice([1, -1])
    return list


def Weight(n, list):
    # 对球按编码进行称重,第i次取第i位编码为0和1的小球并分别置于左右两盘，
    # 即每次称重左盘均表示0，右盘为1，根据每次称重天平的结果（轻、重、平
    # 衡）添加对应编码
    model = []  # 生成标准球列表
    light = heavy = ''
    print("第1次称重结果为: ")
    left = right = 0  # 每次称重前将左右两盘清空
    fig_left = fig_right = ''  # 编号清空

    for j in range(len(list)):
        # 遍历每一个小球
        if list[j][2][0] == '0':
            # 判断第j个小球编码的第1位是否为0，若是将质量加于左盘，记录编号
            left += list[j][1]
            fig_left += list[j][0]
        elif list[j][2][0] == '1':
            # 判断第j个小球编码的第1位是否为1，若是将质量加于右盘，记录编号
            right += list[j][1]
            fig_right += list[j][0]
        else:
            continue

    if left == right:
        # 若天平平衡，轻编码添加2，重编码添加2，打印称重结果，将盘中所有球
        # 设为标准球
        light += '2'
        heavy += '2'
        print(fig_left, "=", fig_right)
        for k in range(len(list)):
            if list[k][2][0] == '0' or list[k][2][0] == '1':
                model.append(list[k])
    elif left < right:
        # 若天平左倾，轻编码添加0，重编码添加1，打印称重结果，将盘外所有球
        # 设为标准球
        light += '0'
        heavy += '1'
        print(fig_left, "<", fig_right)
        for k in range(len(list)):
            if list[k][2][0] == '2':
                model.append(list[k])
    else:
        # 若天平右倾，轻编码添加1，重编码添加0，打印称重结果，将盘外所有球
        # 设为标准球
        light += '1'
        heavy += '0'
        print(fig_left, ">", fig_right)
        for k in range(len(list)):
            if list[k][2][0] == '2':
                model.append(list[k])
    print('轻球编码为：' + light, '或重球编码为：' + heavy)

    for i in range(1, n):
        ball_temp = []
        print("\n第%d次称重结果为: " % (i + 1))
        left = right = 0  # 每次称重前将左右两盘清空
        count_l = count_r = 0  # 用于对盘中小球计数
        fig_left = fig_right = ''  # 清空编号

        for j in range(len(list)):
            # 遍历每一个小球
            if list[j][2][i] == '0':
                # 判断第j个小球编码的第i位是否为0，若是将质量加于左盘，记录编号
                # 左盘球数加一
                ball_temp.append(list[j])
                left += list[j][1]
                fig_left += list[j][0]
                count_l += 1

            elif list[j][2][i] == '1':
                # 判断第j个小球编码的第i位是否为1，若是将质量加于右盘，记录编号
                # 右盘球数加一
                ball_temp.append(list[j])
                right += list[j][1]
                fig_right += list[j][0]
                count_r += 1

            elif (list[j][2][i] == '2'):
                continue

        if count_l < count_r:
            # 若右盘球数多余左盘，则将一个标准球加于左盘
            for k in model:
                # 检查标准球是否已在盘中
                if k not in ball_temp:
                    left += k[1]
                    fig_left += k[0]
                    break
        elif count_l > count_r:
            for k in model:
                if k not in ball_temp:
                    right += k[1]
                    fig_right += k[0]
                    break

        if left < right:
            # 若天平左倾，轻编码添加0，重编码添加1，打印称重结果
            light += '0'
            heavy += '1'
            print(fig_left, "<", fig_right)
        elif left > right:
            # 若天平右倾，轻编码添加1，重编码添加0，打印称重结果
            light += '1'
            heavy += '0'
            print(fig_left, ">", fig_right)
        elif left == right:
            # 若天平平衡，轻编码添加2，重编码添加2，打印称重结果
            light += '2'
            heavy += '2'
            print(fig_left, "=", fig_right)
        print('轻球编码为：' + light, '或重球编码为：' + heavy)
    return light, heavy


def CompareWeight(light, heavy, list):
    # 根据轻重编码查找对应小球
    n = len(list)
    for i in range(n):
        if light in list[i]:
            print("\n算法结果：球 %s 较轻！" % (list[i][0]))
        if heavy in list[i]:
            print("\n算法结果：球 %s 较重！" % (list[i][0]))


def PrintWeight(list):
    # 打印所有小球编号及其质量，用于对比实验结果
    print()
    print("各球实际的质量为：")
    n = len(list)
    for i in range(n):
        print(list[i][0], ":", list[i][1])


if __name__ == '__main__':
    # 输入球数，并计算所需称重次数,球数除以三所得余数
    n_times, n_balls, n_remain = InputBallnum()
    # 根据称重次数产生一个编码序列
    Star_Code = StarCode(n_times)
    Fina_Code = FinalCode(n_balls, Star_Code, n_remain)
    # 根据以上编码序列生成一个表示小球的列表，元素包含小球编码、质量、标号
    init_ball = SetBall(n_balls, Fina_Code, n_remain)
    # 改变一个小球的质量，手动输入或随机生成
    ball = ChangeWeight(init_ball)
    # 根据编码逐次称重、返回称重结果的编码
    light, heavy = Weight(n_times, ball)
    # 将结果编码与小球编码进行对比，找出质量不同的小球，打印其编号
    CompareWeight(light, heavy, ball)
    # 打印所有小球编号及其质量
    PrintWeight(ball)



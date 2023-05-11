list = [[i for i in range(101)] for _ in range(2)]

with open("/lab/tmpig8e/u/yuecheng/yuecheng_log/101_sample/total.txt") as f:
    for line in f.readlines()[1:]:
        task_detail = line.strip().split(",")
        if float(task_detail[-1]) == 0:
            one = 0
        elif float(task_detail[-2]) == 0:
            one = 1
        two = int(task_detail[0])
        list[one][two] = -1
for lis in list:
    for li in lis:
        if li != -1 and li not in [47,56,84,93]:
            print(li)

    print("------------------------------------------------")

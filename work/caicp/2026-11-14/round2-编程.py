n = int(input())
tot = {}  # 站点名集合
order = []  # 排序
for epoch in range(n):
    line = input().split()  # 读入整行
    station, rain = line[0], int(line[1])  # 分别读入站点名和降雨量
    if station not in tot:  # 统计不在站点名中的站点，并统计入order
        tot[station] = 0
        order.append(station)
    tot[station] += rain

rnk = sorted(order, key=lambda s: -tot[s])  # 以order为键值，降序排序
for i, station in enumerate(rnk, 1):
    print(i, station, tot[station])  # 输出

# 以上为27题，接下来是28题

n = int(input())
a = list(map(int, input().split()))
empt = sum(1 for x in a if x == -1)

last = None
left = [None] * n
for i, x in enumerate(a):
    if x != -1:
        last = x
    left[i] = last

right_first = next(x for x in a if x != -1)
filled = [right_first if left[i] is None else left[i] for i in range(n)]

print(min(filled), max(filled), round(sum(filled) / n, 2), empt)

# 27题
n = int(input())
scores = list(map(int, input().split()))
A, B, C, D = 0, 0, 0, 0
for a in scores:
    if a >= 90 and a <= 100:
        A += 1
    if a >= 80 and a <= 89:
        B += 1
    if a >= 70 and a <= 79:
        C += 1
    if a >= 0 and a <= 69:
        D += 1
print(A, B, C, D)

# 28题
n, t = map(int, input().split())
a = list(map(int, input().split()))
ans = -1
for i in range(n - 2):
    if a[i] < t and a[i + 1] < t and a[i + 2] < t:
        ans = i + 1
        break
print(ans)

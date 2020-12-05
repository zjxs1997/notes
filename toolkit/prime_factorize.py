# 分解质因数

num = int(input())

bool_prime_list = [True for _ in range(num + 1)]

for n in range(2, num + 1):
    if not bool_prime_list[n]:
        continue
    i = 2
    while n * i <= num:
        bool_prime_list[n * i] = False
        i += 1

factor = 2
factorize_dict = {}

while not bool_prime_list[num]:
    if num % factor == 0:
        if factor in factorize_dict:
            factorize_dict[factor] += 1
        else:
            factorize_dict[factor] = 1
        num = num // factor
    else:
        factor += 1
        while not bool_prime_list[factor]:
            factor += 1

if num in factorize_dict:
    factorize_dict[num] += 1
else:
    factorize_dict[num] = 1
    
print(factorize_dict)


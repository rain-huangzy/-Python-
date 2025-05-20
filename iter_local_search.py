import random
import math

CITY_SIZE = 52  # 城市数量

# 城市坐标
city = tuple[int, int]202505190523.py
CITIES = list[city]


# 优化值
Delta: list[list[float]] = []

# 解决方案
class Solution:
    def __init__(self):
        self.permutation: list[int] = [0] * CITY_SIZE  # 城市排列
        self.cost: float = 0.0  # 该排列对应的总路线长度


# 计算两个城市间距离
def distance_2city(c1: city, c2: city) -> float:
    distance = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return distance


# 根据产生的城市序列，计算旅游总距离
def cost_total(cities_permutation: list[int], cities: CITIES) -> float:
    total_distance = 0
    for i in range(CITY_SIZE):
        c1 = cities_permutation[i]
        if i == CITY_SIZE - 1:  # 最后一个城市和第一个城市计算距离
            c2 = cities_permutation[0]
        else:
            c2 = cities_permutation[i + 1]
        total_distance += distance_2city(cities[c1], cities[c2])
    return total_distance


# 获取随机城市排列, 用于产生初始解
def random_permutation(cities_permutation: list[int]):
    temp = list(range(CITY_SIZE))
    for i in range(CITY_SIZE - 1):
        r = random.randint(0, len(temp) - 1)
        cities_permutation[i] = temp[r]
        temp[r] = temp[-1]
        temp.pop()
    cities_permutation[CITY_SIZE - 1] = temp[0]


# 颠倒数组中下标begin到end的元素位置, 用于two_opt邻域动作
def swap_element(p: list[int], begin: int, end: int):
    while begin < end:
        p[begin], p[end] = p[end], p[begin]
        begin += 1
        end -= 1


# 邻域动作 反转index_i <-> index_j 间的元素
def two_opt_swap(cities_permutation: list[int], new_cities_permutation: list[int], index_i: int, index_j: int):
    new_cities_permutation[:] = cities_permutation[:]
    swap_element(new_cities_permutation, index_i, index_j)


# 计算邻域操作优化值
def calc_delta(i: int, k: int, tmp: list[int], cities: CITIES) -> float:
    if (i == 0) and (k == CITY_SIZE - 1):
        delta = 0
    else:
        i2 = (i - 1 + CITY_SIZE) % CITY_SIZE
        k2 = (k + 1) % CITY_SIZE
        delta = -distance_2city(cities[tmp[i2]], cities[tmp[i]]) + \
                distance_2city(cities[tmp[i2]], cities[tmp[k]]) - \
                distance_2city(cities[tmp[k]], cities[tmp[k2]]) + \
                distance_2city(cities[tmp[i]], cities[tmp[k2]])
    return delta


# 更新Delta
def Update(i: int, k: int, tmp: list[int], cities: CITIES):
    if i and k != CITY_SIZE - 1:
        i -= 1
        k += 1
        for j in range(i, k + 1):
            for l in range(j + 1, CITY_SIZE):
                Delta[j][l] = calc_delta(j, l, tmp, cities)
        for j in range(0, k):
            for l in range(i, k + 1):
                if j >= l:
                    continue
                Delta[j][l] = calc_delta(j, l, tmp, cities)
    else:
        for i in range(CITY_SIZE - 1):
            for k in range(i + 1, CITY_SIZE):
                Delta[i][k] = calc_delta(i, k, tmp, cities)


# 本地局部搜索，边界条件 max_no_improve
def local_search(best_solution: Solution, cities: CITIES, max_no_improve: int):
    count = 0
    inital_cost = best_solution.cost  # 初始花费
    now_cost = 0
    current_solution = Solution()

    for i in range(CITY_SIZE - 1):
        for k in range(i + 1, CITY_SIZE):
            Delta[i][k] = calc_delta(i, k, best_solution.permutation, cities)

    while count <= max_no_improve:
        # 枚举排列
        for i in range(CITY_SIZE - 1):
            for k in range(i + 1, CITY_SIZE):
                # 邻域动作
                two_opt_swap(best_solution.permutation, current_solution.permutation, i, k)
                now_cost = inital_cost + Delta[i][k]
                current_solution.cost = now_cost
                if current_solution.cost < best_solution.cost:
                    count = 0  # better cost found, so reset
                    best_solution.permutation = current_solution.permutation.copy()
                    best_solution.cost = current_solution.cost
                    inital_cost = best_solution.cost
                    Update(i, k, best_solution.permutation, cities)
        count += 1


# 判断接受准则
def AcceptanceCriterion(cities_permutation: list[int], new_cities_permutation: list[int], cities: CITIES) -> bool:
    AcceptLimite = 500
    c1 = cost_total(cities_permutation, cities)
    c2 = cost_total(new_cities_permutation, cities) - 500
    if c1 < c2:
        return False
    else:
        return True


# 将城市序列分成4块，然后按块重新打乱顺序。用于扰动函数
def double_bridge_move(cities_permutation: list[int], new_cities_permutation: list[int], cities: CITIES):
    pos = [0]
    pos.append(random.randint(1, CITY_SIZE // 3))
    pos.append(random.randint(CITY_SIZE // 3 + 1, 2 * CITY_SIZE // 3))
    pos.append(random.randint(2 * CITY_SIZE // 3 + 1, CITY_SIZE - 1))
    pos.append(CITY_SIZE)

    random_order = list(range(4))
    random.shuffle(random_order)

    deadprotect = 0
    while True:
        i = 0
        for j in range(4):
            for k in range(pos[random_order[j]], pos[random_order[j] + 1]):
                new_cities_permutation[i] = cities_permutation[k]
                i += 1
        deadprotect += 1
        if deadprotect == 5 or not AcceptanceCriterion(cities_permutation, new_cities_permutation, cities):
            break


# 扰动
def perturbation(cities: CITIES, best_solution: Solution, current_solution: Solution):
    double_bridge_move(best_solution.permutation, current_solution.permutation, cities)
    current_solution.cost = cost_total(current_solution.permutation, cities)


# 迭代搜索
def iterated_local_search(best_solution: Solution, cities: CITIES, max_iterations: int, max_no_improve: int):
    current_solution = Solution()
    # 获得初始随机解
    random_permutation(best_solution.permutation)
    best_solution.cost = cost_total(best_solution.permutation, cities)
    local_search(best_solution, cities, max_no_improve)  # 初始搜索

    for i in range(max_iterations):
        perturbation(cities, best_solution, current_solution)  # 扰动+判断是否接受新解
        local_search(current_solution, cities, max_no_improve)  # 继续局部搜索
        # 找到更优解
        if current_solution.cost < best_solution.cost:
            best_solution.permutation = current_solution.permutation.copy()
            best_solution.cost = current_solution.cost
        print(f"迭代搜索 {i:13} 次\t最优解 = {best_solution.cost} 当前解 = {current_solution.cost}")


# berlin52城市坐标，最优解7542好像
berlin52: CITIES = [(565, 575), (25, 185), (345, 750), (945, 685), (845, 655),
                     (880, 660), (25, 230), (525, 1000), (580, 1175), (650, 1130), (1605, 620),
                     (1220, 580), (1465, 200), (1530, 5), (845, 680), (725, 370), (145, 665),
                     (415, 635), (510, 875), (560, 365), (300, 465), (520, 585), (480, 415),
                     (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400), (660, 180),
                     (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
                     (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960),
                     (95, 260), (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65),
                     (830, 610), (605, 625), (595, 360), (1340, 725), (1740, 245)]


def main():
    global Delta
    random.seed(1)
    max_iterations = 600
    max_no_improve = 50
    # 初始化指针数组
    Delta = [[0] * CITY_SIZE for _ in range(CITY_SIZE)]

    best_solution = Solution()
    iterated_local_search(best_solution, berlin52, max_iterations, max_no_improve)

    print("\n\n搜索完成！最优路线总长度 =", best_solution.cost)
    print("最优访问城市序列如下：")
    for i in range(CITY_SIZE):
        print(f"{best_solution.permutation[i]:4}", end='')
    print("\n\n")


if __name__ == "__main__":
    main()
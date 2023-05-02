import random,math,statistics
from typing import List, Tuple
import scipy


def cities_gen(cities_num=20,maximum_x=500,maximum_y=500):
    cities=[]
    for i in range(cities_num):
        point = (random.randint(0,maximum_x),random.randint(0,maximum_y))
        while point in cities:
            point = (random.randint(0,maximum_x),random.randint(0,maximum_y))
        cities.append(point)
    return cities

def Jaccard_相似性():
    pass

def distance_matrix_gen(cities: List[Tuple[int, int]]) -> List[List[float]]:

    return [[math.dist(c1,c2) for c2 in cities] for c1 in cities]

def fitness(individual: List[int], dist_matrix: List[List[float]]) -> float:
    r = sum(dist_matrix[individual[i]][individual[i + 1]] for i in range(len(individual) - 1)) + dist_matrix[individual[0]][individual[-1]]
    return r

def selection(population: List[List[int]], dist_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    candidates = random.sample(population, 4)
    candidates.sort(key=lambda x: fitness(x, dist_matrix))
    return candidates[0], candidates[1]

def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]

    idx = end
    for i in range(len(parent1)):
        if parent2[(i + end) % len(parent1)] not in child:
            child[idx] = parent2[(i + end) % len(parent1)]
            idx = (idx + 1) % len(parent1)

    return child

def mutate(individual: List[int]) -> List[int]:
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def ga_tsp(mut_rate,cities: List[Tuple[int, int]], population_size: int, generations: int) -> Tuple[List[int], float]:
    dist_matrix = distance_matrix_gen(cities)
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(population_size)]
    # 我要稳定实现 800左右的最小值
    history_best_score = 10000000
    history_score_variance = []
    for generation in range(generations):
        # 数据处理, 获得种群分数, 最高,最低分,平均分,历史最佳分,
        # 新种群生成规则: 从父代中按规则选出若干个体, 这些个体一并加入新种群, 如果种群未满, 则从这些父类个体中任选两个交叉繁殖新一代, 直到种群满为止
        # 前期, 保持种群多样性,增加搜索空间,
        #   选择: 按照适应度排序,选偶数位置的个体作为复制对象,
        #   交叉: 常规操作
        #   变异: 不同个体的差异应变大, 比平均值成绩差的减缓变异率,比平均值成绩好的加速变异率
        # 中期, 保持种群稳定性,维持局部最优,
        #   选择: 按照适应度排序,选偶数位置的个体作为复制对象,
        #   交叉: 常规操作
        #   变异: 应在各个稳定低谷保持一定的聚集数量, 首先计算各个序列之间的相似度, 再对相似度用密度聚类方法实现, 边缘变异率高, 中心变异率低
        # 后期, 种群加速收敛, 直接使用锦标赛规则,
        #   选择: 直接选最优的
        #   交叉: 常规操作
        #   变异:
        fitness_scores = [fitness(individual, dist_matrix) for individual in population]
        min_score, max_score,mean_score = min(fitness_scores), max(fitness_scores),statistics.mean(fitness_scores)
        history_best_score = min(min_score, history_best_score)
        new_population = []

        # 选择

        # 交叉

        # 变异
        population = new_population

    best_individual = min(population, key=lambda x: fitness(x, dist_matrix))
    return best_individual, fitness(best_individual, dist_matrix)

if __name__ == "__main__":
    cities = cities_gen()
    population_size = 100
    generations = 1000
    mut_rate = 0.15
    best_individual, best_distance = ga_tsp(mut_rate,cities, population_size, generations)
    print("最优路径:", best_individual)
    print("最短距离:", best_distance)
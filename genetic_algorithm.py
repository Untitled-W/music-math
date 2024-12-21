import random
import numpy as np
import musicpy as mp
from musicpy import *
from read_data import readBars, readMainTracks
import re
from collections import Counter

class GeneticAlgorithm:
    def __init__(
        self,
        population,
        fitness,
        mutation_rate,
        crossover_rate,
        selection_rate,
        generations,
    ):
        self.mutation_rate = mutation_rate
        # Probability of mutation: which is the probability that a gene will change its value
        self.crossover_rate = crossover_rate
        # Probability of crossover: which is the probability that two chromosomes will exchange genetic information
        self.selection_rate = selection_rate
        # Selection rate: which is the percentage of the population that will be selected for crossover
        self.generations = generations  # Number of generations
        self.population = population
        self.fitness = fitness

        # 定义类中的调性
        self.key_name = None
        self.scale_type = None

        # 定义音程质量评估（不和谐度）
        # self.interval_values = [1, 3, 2, 1, 1, 2, 3, 1] 
        self.interval_values = [1, 6, 4, 1, 1, 4, 6, 1] # 增大音程和谐度差异 

        # 添加初始种群的音程均值和方差属性
        self.initial_mean = None
        self.initial_variance = None

        # 初始化均值和方差
        self._initialize_population_statistics()

        # 确定调性内的音符
        self.tonality_notes = scale(self.key_name, self.scale_type).notes

    def _initialize_population_statistics(self):
        """计算初始种群的音程均值和方差"""
        all_tonalities = []
        all_intervals = []
        for chord in self.population:
            # 提取每个个体的音程
            current_scales = mp.alg.detect_scale3(chord, key_accuracy_tol=0.8)
            all_tonalities.extend(re.findall(r'([A-G][#b]? major)', current_scales))
            intervals = chord.intervalof(translate=True, cumulative=False)
            for interval in intervals:
                if interval.number < 9:
                    all_intervals.append(self.interval_values[interval.number - 1])
                else:
                    all_intervals.append(5)
        counter = Counter(all_tonalities)
        most_common_major = counter.most_common(1)[0]
        print(f"初始种群中出现次数最多的调是: {most_common_major[0]}，占比为: {most_common_major[1]/len(all_tonalities)}")
        self.key_name = most_common_major[0].split()[0]
        self.scale_type = most_common_major[0].split()[1]

        self.initial_mean = np.mean(all_intervals)
        self.initial_variance = np.var(all_intervals)

        # # 将初始种群的所有chord都转化为指定的调性
        # revised_population = []
        # for chord in self.population:
        #     current_scales = mp.alg.detect_scale3(chord, key_accuracy_tol=0.8)
        #     tonality = re.findall(r'([A-G][#b]? major)', current_scales)
        #     new_chord = chord
        #     if tonality != most_common_major:
        #         new_chord = chord.modulation(scale(tonality[0].split()[0], tonality[0].split()[1]), scale(most_common_major[0].split()[0], most_common_major[0].split()[1]))
        #     revised_population.append(new_chord)
        # self.population = revised_population

        # for chord in self.population:
        #     current_scales = mp.alg.detect_scale3(chord, key_accuracy_tol=0.8)
        #     tonality = re.findall(r'([A-G][#b]? major)', current_scales)
        #     print(tonality)
        # import sys
        # sys.exit()

    def _selection(self):
        original_weights = []
        for chord in self.population:
            value = self.fitness(self.interval_values, self.initial_mean, self.initial_variance, self.tonality_notes, chord)
            if math.isnan(value):
                print(chord)
                raise ValueError("Fitness function returns NaN")
            original_weights.append(value)
        # fitness函数是设计的值越小越好，所以这里要取倒数
        inverted_weights = [1 / weight if weight != 0 else float('inf') for weight in original_weights]
        # inverted_weights = [20 -  weight if weight != 0 else float('inf') for weight in original_weights]
        
        selected = random.choices(
            self.population,
            k=int(self.selection_rate * len(self.population)),
            weights=inverted_weights,
        )
        # filted_selected = [chord for chord in selected if len(chord.notes) >= 5 and len(chord.notes) <= 70]
        filted_selected = selected
        return filted_selected

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.random() * parent1.bars()
            return parent1.cut(0, point) + parent2.cut(point, parent2.bars()), \
                   parent2.cut(0, point) + parent1.cut(point, parent1.bars())
        return parent1, parent2

    def _mutate(self, chord):
        for i in range(len(chord.notes)):
            if random.random() < self.mutation_rate:
                chord.notes[i] += random.choice([-1, 1])
        return chord

    def _create_new_population(self, selected):
        new_population = []
        for i in range(0, len(selected)-1, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            offspring1, offspring2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(offspring1))
            new_population.append(self._mutate(offspring2))
        return new_population

    def run(self):
        from tqdm import tqdm
        for generation in range(self.generations):
            selected = self._selection()
            self.population = self._create_new_population(selected)
            best_fitness = min(self.fitness(self.interval_values, self.initial_mean, self.initial_variance, self.tonality_notes, chord) for chord in self.population)
            best_genome = self.population[np.argmin([self.fitness(self.interval_values, self.initial_mean, self.initial_variance, self.tonality_notes, chord) for chord in self.population])]
            print(f"Epoch = {generation}; Best Fitness = {best_fitness}; Population Size = {len(self.population)}")
            if generation == self.generations - 1:
                write(best_genome, name="best_1.mid", save_as_file=True)
            yield best_genome, best_fitness

def _initialize_population():
    # syf
    """
    should return a list. every element is a musicpy's chord object,which looks like:
        chord(notes=[C4, E4, G4, C4, E4, G4, C4, E4, G4, E3, ...], interval=[0, 0, 1/2, 0, 0, 1/2, 0, 0, 1/2, 1/8, ...], start_time=0)
    """
    # 如果以小节作为population，用readBars；如果用主旋律做population，用readMainTracks
    '''
    chords = readMainTracks()
    return chords
    '''
    BAR_MIN_LENGTH = 5
    BAR_MAX_LENGTH = 10
    bars = readBars()
    filtered_bars = []
    for bar in bars:
        if len(bar) < BAR_MIN_LENGTH or len(bar) > BAR_MAX_LENGTH or len(bar.notes) < 10 or len(bar.notes) > 60:
            continue
        filtered_bars.append(bar)
    return filtered_bars


def fitness(interval_values, initial_mean, initial_variance, tonality_notes, chord):
    # hcy
    """
    input a chord (with the form above), output a float number, which is the fitness of this sequence.
    """
    """
    计算输入和弦的适应度：
    1. 音程质量（不和谐度）。
    2. 计算调性外音符。
    3. 比较与参考和弦（初始种群）的均值和方差。
    """
    # 超参
    alpha, beta, gama, delta, theta = 1.0, 1.0, 1.0, 100, 1

    # 如果chord的note只有一个，直接返回一个较大的惩罚
    if len(chord.notes) < 2:
        return 100

    fitness_score = 0

    # 提取和弦的音程
    intervals = chord.intervalof(translate=True, cumulative=False)

    values = []
    for interval in intervals:
        if interval.number < 9:
            values.append(interval_values[interval.number - 1])
        else:
            values.append(5)
    mean = np.mean(values)
    variance = np.var(values)

    # 将均值差异和方差差异加入到适应度中
    fitness_score += abs(mean - initial_mean) * alpha
    fitness_score += abs(variance - initial_variance) * beta

    # 统计调性外音符
    out_of_tonality_notes_num = 0
    for note in chord.notes:
        flag = False
        for tonality_note in tonality_notes:
            if (note.degree - tonality_note.degree) % 12 == 0:
                flag = True
                break
        if not flag:
            out_of_tonality_notes_num += 1
    fitness_score += out_of_tonality_notes_num * gama

    # 新添加一个惩罚音符太少的项
    fitness_score += 1/(len(chord.notes)) * delta

    # 新添加一个惩罚不和谐音程的项
    fitness_score += mean * theta

    # 返回适应度分数（分数越低适应度越高）
    return fitness_score


if __name__ == "__main__":
    music_piece = _initialize_population()
    ga = GeneticAlgorithm(
        population=music_piece,
        fitness=fitness,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_rate=1.0,
        generations=100,
    )
    result = ga.run()
    r_best_genome = mp.chord([])
    with open("result.txt", "w") as f:
        for best_genome, best_fitness in result:
            r_best_genome = r_best_genome + best_genome[:-1]
            f.write(f"{best_genome}\t{best_fitness}\n")
    
    mp.write(r_best_genome, name='best.mid')
from random import choice, randint, seed, choices
from pprint import pprint
import numpy as np
from copy import deepcopy
import heapq
from pickle import dump
#seed(42)


class Room:

    def __init__(self, x, y, _len, width, _type):
        self.x = x
        self.y = y
        self._len = _len
        self.width = width
        self._type = _type

    def get_area(self):
        return self._len * self.width

    def __str__(self):
        return str((self.x, self.y, self._len, self.width, self._type))

    def __repr__(self):
        return str((self.x, self.y, self._len, self.width, self._type))


class Building:

    def __init__(self, rooms: list):
        self.rooms = rooms
        self.area = 0
        self.update_area()
        self.score = 0
        self.basic_sores = [
            self.maximize_rooms,
            self.maximize_total_rooms_area,
            self.minimize_total_rooms_area,
        ]

    def __getitem__(self, index):
        return self.rooms[index]

    def __setitem__(self, key, value):
        self.rooms[key] = value

    def __str__(self):
        return '[' + ', '.join(map(str, self.rooms)) + ']'

    def __repr__(self):
        return '[' + ', '.join(map(str, self.rooms)) + ']'

    def update_score(self):
        new_score = 1
        for func in self.basic_sores:
            new_score *= func()
        assert isinstance(new_score, int)
        self.score = new_score

    def get_score(self):
        self.update_score()
        return self.score

    def update_area(self):
        assert len(self.rooms) > 0
        self.area = sum(room.get_area() for room in self.rooms)

    def get_area(self):
        # assert self.rooms[0] is Room
        return self.area

    # Evaluation methods
    def maximize_rooms(self):
        return len(self.rooms)

    def maximize_total_rooms_area(self):
        return self.get_area()

    def minimize_total_rooms_area(self):
        return 1000 * len(self.rooms) - self.get_area()

    # Graph evaluation(Deprecated)


class Level:
    types = ['o', 'u']

    def __init__(self, n=1, rooms_count=1, max_coord=1, max_value=10, parents_count=3):
        self.population = Level.generate_population(n,
                                                    rooms_count,
                                                    max_coord,
                                                    max_value)
        self.rooms_count = rooms_count
        self.n = n
        self.population_rank = []
        for i in range(self.n):
            heapq.heappush(self.population_rank, (self.population[i].get_score(), i))
        self.population_indexes = [i for i in range(self.n)]
        self.parents_count = parents_count

    def inherit_gene(self, building1, building2):
        distributed = np.random.normal(0.5, 0.5, self.rooms_count)
        isnt_switched = True
        for i in range(self.rooms_count):
            if distributed[i] > 0.5:
                building2[i] = deepcopy(building1[i])
                isnt_switched = False
        if isnt_switched:
            i = randint(0, self.rooms_count - 1)
            building2[i] = deepcopy(building1[i])

    def select_mating_pool(self, pool_count) -> list:
        pool_indexes = choices(self.population_indexes, k=pool_count)
        return [self.population[i] for i in pool_indexes]

    def kill_worst(self):
        self.population = [self.population[item[1]]
                           for item in heapq.nlargest(self.n, self.population_rank)]
        largest = heapq.nlargest(self.n, self.population_rank)
        self.population_rank = []
        for i, building in enumerate(largest):
            heapq.heappush(self.population_rank, (building[0], i))

    def make_iteration(self):
        current_building = None
        for i in range(self.n):
            current_building = deepcopy(self.population[i])
            other = self.select_mating_pool(self.parents_count)#this parameter you can change
            for parent in other:
                if id(self.population[i]) != id(parent):
                    self.inherit_gene(parent, current_building)
            current_index = len(self.population)
            self.population.append(current_building)
            heapq.heappush(self.population_rank, (current_building.get_score(), current_index))
        self.kill_worst()

    @staticmethod
    def generate_population(n, rooms_count, max_coord, max_value):
        return [Building([Level.generate_item(max_coord, max_value)
                          for _ in range(rooms_count)])
                for _ in range(n)]

    @staticmethod
    def generate_item(max_coord, max_value):
        x = randint(1, max_coord)
        y = randint(1, max_coord)
        _len = randint(1, max_value)
        width = randint(1, max_value)
        _type = choice(Level.types)
        return Room(x, y, _len, width, _type)




def main():
    test_level = Level(100, 10, 10, 15, 5)
    for _ in range(100):
        test_level.make_iteration()
    with open('first_generated.pkl', 'wb') as fout:
        dump(test_level.population, fout)
    # pprint(test_level.population)


if __name__ == "__main__":
    main()

from random import choice, randint, seed, choices
from pprint import pprint
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
import heapq
from pickle import dump
import pytest
import os


# seed(42)


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

    # Graph evaluation

    @staticmethod
    def is_crossovered_by_axis(a1, b1, a2, b2):
        a1, a2 = min(a1, a2), max(a1, a2)
        b1, b2 = min(b1, b2), max(b1, b2)
        a = max(a1, a2)
        b = min(b1, b2)
        return b - a >= 0

    @staticmethod
    def point_equality(a1, len1, a2, len2, a):
        return a1 + a2 - 2 * a == len1 or a1 + a2 - 2 * a == len2

    def has_common_dot(self, other_room):
        assert isinstance(other_room, Room)
        x = min(self.x, other_room.x)
        y = min(self.y, other_room.y)
        if self.point_equality(self.x, self.width,
                               other_room.x, other_room.width, x) and \
                self.point_equality(self.y, self._len,
                                    other_room.y, other_room._len, y):
            return True
        return False

    def is_crossovered(self, other_room):
        assert isinstance(other_room, Room)
        if self.is_crossovered_by_axis(self.x,
                                       self.x + self.width,
                                       other_room.x,
                                       other_room.x + other_room.width) and \
                self.is_crossovered_by_axis(self.y,
                                            self.y + self._len,
                                            other_room.y,
                                            other_room.y + other_room._len):
            if self.has_common_dot(other_room):
                return False
            return True
        return False


class Building:

    def __init__(self, rooms: list):
        self.width = 70
        self.height = 60
        self.teil_size = 16
        self.rooms = rooms
        self.area = 0
        self.update_area()
        self.score = 0
        self.basic_sores = [
            self.maximize_rooms,
            self.maximize_total_rooms_area,
            self.minimize_total_rooms_area,
        ]
        self.graph = []

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

    def prepare_building(self):
        self.graph = [set()] * len(self.rooms)
        for i in range(len(self.rooms)):
            for j in range(len(self.rooms)):
                if i != j:
                    if self.rooms[i].is_crossovered(self.rooms[j]):
                        self.graph[i].add(j)

    def create_image(self, filename):
        img = Image.new("RGB", (self.width * self.teil_size, self.height * self.teil_size))
        img1 = ImageDraw.Draw(img)
        for idx, room in enumerate(self.rooms):
            x, y, w, l = map(lambda x: x * self.teil_size,
                             (room.x, room.y, room.width, room._len))
            shape = [(x, y), (x + w, y + l)]
            img1.rectangle(shape, outline="#800080")
            img1.text((x + 5, y + 5), str(idx))
        path = os.getcwd() + '/' + filename
        img.save(path)


class Level:
    types = ['o', 'u']

    def __init__(self, n=1, rooms_count=1,
                 max_coord=1, max_value=10, parents_count=3, min_value=6):
        self.population = Level.generate_population(n,
                                                    rooms_count,
                                                    max_coord,
                                                    max_value,
                                                    min_value)
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
        for i in range(self.n):
            current_building = deepcopy(self.population[i])
            other = self.select_mating_pool(self.parents_count)  # this parameter you can change
            for parent in other:
                if id(self.population[i]) != id(parent):
                    self.inherit_gene(parent, current_building)
            current_index = len(self.population)
            self.population.append(current_building)
            heapq.heappush(self.population_rank, (current_building.get_score(), current_index))
        self.kill_worst()

    @staticmethod
    def generate_population(n, rooms_count, max_coord, max_value, min_value):
        return [Building([Level.generate_item(max_coord, max_value, min_value)
                          for _ in range(rooms_count)])
                for _ in range(n)]

    @staticmethod
    def generate_item(max_coord, max_value, min_value):
        x = randint(1, max_coord)
        y = randint(1, max_coord)
        _len = randint(min_value, max_value)
        width = randint(min_value, max_value)
        _type = choice(Level.types)
        return Room(x, y, _len, width, _type)

    def fit(self, rounds=1):
        for _ in range(rounds):
            self.make_iteration()


def main():
    test_level = Level(n=5, rooms_count=10, max_coord=40, max_value=25, min_value=6)
    test_level.fit(20)
    test_level.population[0].prepare_building()
    test_level.population[0].create_image("graph_implementing.png")
    pprint(test_level.population[0].graph)
    # with open('offered_generated.pkl', 'wb') as fout:
    #     dump(test_level.population, fout)


def test_is_crossovered_by_axis1():
    a = Room(0, 0, 7, 7, 'o')
    assert not a.is_crossovered_by_axis(0, 1, 2, 3)


def test_is_crossovered_by_axis2():
    a = Room(0, 0, 7, 7, 'o')
    assert a.is_crossovered_by_axis(0, 1, 1, 3)


def test_is_crossovered_by_axis3():
    a = Room(0, 0, 7, 7, 'o')
    assert a.is_crossovered_by_axis(0, 2, 1, 3)


def test_is_crossovered_by_axis4():
    a = Room(0, 0, 7, 7, 'o')
    assert a.is_crossovered_by_axis(0, 1, 0, 3)


def test_is_crossovered1():
    a = Room(0, 0, 7, 7, 'o')
    b = Room(0, 0, 7, 7, 'u')
    assert a.is_crossovered(b)


def test_is_crossovered2():
    a = Room(0, 0, 7, 7, 'o')
    b = Room(7, 7, 7, 7, 'u')
    assert not a.is_crossovered(b)


def test_is_crossovered3():
    a = Room(0, 0, 7, 7, 'o')
    b = Room(3, 3, 7, 7, 'u')
    assert a.is_crossovered(b)


if __name__ == "__main__":
    main()

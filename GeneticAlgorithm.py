from random import choice, randint, seed, choices
from pprint import pprint
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
import heapq
from pickle import dump
import pytest
import os
from collections import deque, namedtuple

np.random.seed(42)
seed(42)


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

    def is_belong(self, other_room):
        assert isinstance(other_room, Room)
        return self.x >= other_room.x and self.x + self.width <= other_room.x + other_room.width \
               and self.y >= other_room.y and self.y + self.width <= other_room.y + other_room.width


class Building:
    fields = ['first_meet', 'second_meet', 'third_meet', 'fourth_meet']

    def __init__(self, rooms: list):
        self.width = 70
        self.height = 60
        self.tile_size = 32
        self.rooms = rooms
        self.area = 0
        self.update_area()
        self.score = 0
        self.basic_sores = [
            self.maximize_rooms,
            self.maximize_total_rooms_area,
            self.minimize_total_rooms_area,
        ]
        self.graph = [{} for _ in range(len(self.rooms))]
        self.map = []
        self.horizontal = []
        self.vertical = []
        self.walls_tile = 0
        self.outdoor_tile = -1
        self.player_position = 608, 175
        self.xml_initialization = f'''<?xml version="1.0" encoding="UTF-8"?>
<map version="1.2" tiledversion="1.3.3" orientation="orthogonal" renderorder="right-down" width="{self.width}" height="{self.height}" tilewidth="{self.tile_size}" tileheight="{self.tile_size}" infinite="0" nextlayerid="9" nextobjectid="410">
 <properties>
  <property name="player position" value="{str(self.player_position[0]) + ', ' + str(self.player_position[1])}"/>
 </properties>
 <tileset firstgid="1" source="rooms.tsx"/>'''
        self.xml_ending = '''</map>'''

    def __getitem__(self, index):
        return self.rooms[index]

    def __setitem__(self, key, value):
        self.rooms[key] = value

    def __str__(self):
        return '\n'.join(
            [','.join(map(str, self.map[i]))
             for i in range(self.height)])

    def __repr__(self):
        return '\n'.join(
            [','.join(map(lambda x: str(x).rjust(2), self.map[i]))
             for i in range(self.height)])

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
        for i in range(len(self.rooms)):
            for j in range(len(self.rooms)):
                if i != j:
                    if self.rooms[i].is_crossovered(self.rooms[j]):
                        if j not in self.graph[i]:
                            self.graph[i][j] = [None] * 4
                            self.graph[j][i] = [None] * 4

    def create_image(self, filename):
        img = Image.new("RGB", (self.width * self.tile_size, self.height * self.tile_size))
        img1 = ImageDraw.Draw(img)
        for idx, room in enumerate(self.rooms):
            x, y, w, l = map(lambda x: x * self.tile_size,
                             (room.x, room.y, room.width, room._len))
            shape = [(x, y), (x + w, y + l)]
            img1.rectangle(shape, outline="#800080")
            img1.text((x + 5, y + 5), str(idx))
        path = os.getcwd() + '/' + filename
        img.save(path)

    def draw_room_by_idx(self, idx):
        assert isinstance(self.rooms[idx], Room)
        room = self.rooms[idx]
        idx += 1
        for i in range(room._len):
            if i == 0 or i == room._len - 1:
                j_seq = range(room.width)
            else:
                j_seq = [0, room.width - 1]
            for j in j_seq:
                if self.map[room.y + i][room.x + j] != self.walls_tile \
                        and self.map[room.y + i][room.x + j] != self.outdoor_tile:
                    other_room_idx = self.map[room.y + i][room.x + j] - 1
                    if other_room_idx not in self.graph[idx]:
                        self.graph[other_room_idx][idx] = [None] * 4
                        self.graph[idx][other_room_idx] = [None] * 4
                    k = 0
                    while k < 4 and self.graph[other_room_idx][idx][k] is None:
                        k += 1
                    if k == 4:
                        pass#if two walls equal
                    else:
                        self.graph[other_room_idx][idx][k] = \
                            self.graph[idx][other_room_idx][k] = i, j
                self.map[room.y + i][room.x + j] = idx

    def in_map_field(self, i, j):
        return i >= 0 and j >= 0 and i < self.height and j < self.width

    def _horisontal_dfs(self, i, j, dy, dx, array, dy_list, dx_list):
        if self.map[i][j] != self.walls_tile \
                and self.map[i][j] != self.outdoor_tile:
            updated_idx = None
            if dy + dx > 0:
                updated_idx = 1
            else:
                updated_idx = 0
            array[updated_idx][0] += dy
            array[updated_idx][1] += dx
            self.map[i][j] = self.walls_tile
            self._horisontal_dfs(i + dy, j + dx, dy, dx, array, dy_list, dx_list)

    def _vertical_dfs(self, i, j, dy, dx, array, dy_list, dx_list):
        if self.map[i][j] != self.walls_tile \
                and self.map[i][j] != self.outdoor_tile:
            updated_idx = None
            if dy + dx > 0:
                updated_idx = 1
            else:
                updated_idx = 0
            array[updated_idx][0] += dy
            array[updated_idx][1] += dx
            self.map[i][j] = self.walls_tile
            self._vertical_dfs(i + dy, j + dx, dy, dx, array, dy_list, dx_list)

    def _call_dfs(self, i, j, dy, dx, idx, array, dy_list, dx_list, function):
        saved_i, saved_j = i, j
        i, j = i + dy, j + dx
        if self.map[i][j] != self.walls_tile \
                and self.map[i][j] != self.outdoor_tile:
            if idx is None:
                idx = len(array)
                if dy + dx > 0:
                    array.append([[saved_i, saved_j], [i, j]])
                else:
                    array.append([[i, j], [saved_i, saved_j]])
            function(saved_i, saved_j, dy, dx, array[idx], dy_list, dx_list)
        return idx

    def find_all_walls(self):
        dy = [0, 0, 1, -1]
        dx = [1, -1, 0, 0]
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] != self.walls_tile \
                        and self.map[i][j] != self.outdoor_tile:
                    horizontal_idx, vertical_idx = None, None
                    for k in range(4):
                        if k < 2:
                            horizontal_idx = self._call_dfs(i, j,
                                                            dy[k],
                                                            dx[k],
                                                            horizontal_idx,
                                                            self.horizontal,
                                                            dy,
                                                            dx,
                                                            self._horisontal_dfs)
                        else:
                            if horizontal_idx is None:
                                vertical_idx = self._call_dfs(i, j,
                                                              dy[k],
                                                              dx[k],
                                                              vertical_idx,
                                                              self.vertical,
                                                              dy,
                                                              dx,
                                                              self._vertical_dfs)

    def fill_room_by_tile_num(self, idx, tile_num):
        room = self.rooms[idx]
        for i in range(1, room._len - 1):
            for j in range(1, room.width - 1):
                if self.map[room.y + i][room.x + j] > idx:
                    self.map[room.y + i][room.x + j] = tile_num

    def create_map(self):
        self.map = [[self.outdoor_tile] * self.width for _ in range(self.height)]
        for i in range(len(self.rooms) - 1, -1, -1):
            self.draw_room_by_idx(i)
        for i in range(len(self.rooms)):
            self.fill_room_by_tile_num(i, -1)

    def converted_layer_to_xml(self, layer, floor_tile=0, wall_tile=319, name="layer"):
        starting = f'''<layer id="1" name="{name}" width="{self.width}" height="{self.height}" locked="1">
  <data encoding="csv">'''
        layer = layer.replace(str(self.walls_tile), str(wall_tile))
        layer = layer.replace(str(self.outdoor_tile), str(floor_tile))
        ending = f'''</data>
 </layer>'''
        return '\n'.join((starting, layer, ending))

    def convert_shapes_to_xml(self):
        starting = '''<objectgroup id="8" name="StaticShapes" locked="1">'''
        shapes = []
        for line in self.horizontal:
            x = line[0][1] * self.tile_size
            y = line[0][0] * self.tile_size
            width = (line[1][1] - line[0][1] - 1) * self.tile_size
            shapes.append(f'''  <object id="" x="{x}" y="{y}" width="{width}" height="{self.tile_size}"/>''')
        for line in self.vertical:
            x = line[0][1] * self.tile_size
            y = line[0][0] * self.tile_size
            height = (line[1][0] - line[0][0] - 1) * self.tile_size
            shapes.append(f'''  <object id="" x="{x}" y="{y}" width="{self.tile_size}" height="{height}"/>''')
        ending = ''' </objectgroup>'''
        return '\n'.join((starting, '\n'.join(shapes), ending))


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
    test_level = Level(n=5, rooms_count=5, max_coord=40, max_value=25, min_value=6)
    test_level.fit(20)
    test_level.population[0].create_map()
    test_level.population[0].find_all_walls()
    print(test_level.population[0])
    # layer = str(test_level.population[0])
    # with open("generated_level.tmx", 'w') as fout:
    #     print(test_level.population[0].xml_initialization, file=fout)
    #     print(test_level.population[0].converted_layer_to_xml(layer, name='floor'), file=fout)
    #     print(test_level.population[0].convert_shapes_to_xml(), file=fout)
    #     print(test_level.population[0].xml_ending, file=fout)
    # test_level.population[0].prepare_building()
    # test_level.population[0].create_image("graph_implementing.png")
    # pprint(test_level.population[0].graph)
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

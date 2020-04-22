from random import choice, randint, seed, choices
from pprint import pprint
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
import heapq
from pickle import dump
import pytest
import os
from collections import deque

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


class Building:

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
        self.doors_positions = []
        self.filled_map = []
        self.graph = []
        self.map = []
        self.horizontal = []
        self.vertical = []
        self.tiles_map_floor = []
        self.tiles_map_walls_doors = []
        self.components = []
        self.walls_tile = 0
        self.outdoor_tile = -1
        self.door_place_tile = -2
        self.door_tile = -3
        self.zero_tile = 0
        self.left_vertical_door_tile = 427 + 1
        self.bottom_horizontal_door_tile = 426 + 1
        self.right_vertical_door_tile = 429 + 1
        self.top_horizontal_door_tile = 428 + 1
        self.top_left_corner = 320 + 1
        self.top_right_corner = 321 + 1
        self.bottom_left_corner = 323 + 1
        self.bottom_right_corner = 322 + 1
        self.top_horizontal_wall = 318 + 1
        self.bottom_horizontal_wall = 319 + 1
        self.left_vertical_wall = 317
        self.right_vertical_wall = 318
        self.floor_tile = 386
        self.outdoor = 228
        self.used_tiles = {
            self.walls_tile,
            self.outdoor_tile,
            self.door_place_tile,
            self.left_vertical_door_tile,
            self.bottom_horizontal_door_tile,
            self.right_vertical_door_tile,
            self.top_horizontal_door_tile,
            self.door_tile,
            self.top_left_corner,
            self.top_right_corner,
            self.bottom_left_corner,
            self.bottom_right_corner,
            self.top_horizontal_wall,
            self.bottom_horizontal_wall,
            self.left_vertical_wall,
            self.right_vertical_wall
        }
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

    def print(self, array):
        print('\n'.join(
            [''.join(map(lambda x: str(x).rjust(3), array[i]))
             for i in range(self.height)]))

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
        self.graph = [set() for _ in range(len(self.rooms))]
        for i in range(len(self.rooms)):
            for j in range(len(self.rooms)):
                if i != j:
                    if self.rooms[i].is_crossovered(self.rooms[j]):
                        self.graph[i].add(j)
                        self.graph[j].add(i)

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

    def fill_room_by_idx(self, idx):
        assert isinstance(self.rooms[idx], Room)
        room = self.rooms[idx]
        idx += 1
        for i in range(room._len):
            for j in range(room.width):
                self.filled_map[room.y + i][room.x + j] = idx

    def create_filled_map(self):
        self.filled_map = [[self.outdoor_tile] * self.width for _ in range(self.height)]
        for i in range(len(self.rooms) - 1, -1, -1):
            self.fill_room_by_idx(i)

    def find_all_door_pos_horizontal(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.filled_map[i][j] == self.walls_tile \
                        and self.in_map_field(i, j - 1) \
                        and self.in_map_field(i, j + 1) \
                        and self.filled_map[i][j - 1] not in self.used_tiles \
                        and self.filled_map[i][j + 1] not in self.used_tiles \
                        and self.filled_map[i][j - 1] != self.filled_map[i][j + 1]:
                    self.filled_map[i][j] = self.door_place_tile

    def find_all_door_pos_vertical(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.filled_map[i][j] == self.walls_tile \
                        and self.in_map_field(i - 1, j) \
                        and self.in_map_field(i + 1, j) \
                        and self.filled_map[i - 1][j] not in self.used_tiles \
                        and self.filled_map[i + 1][j] not in self.used_tiles \
                        and self.filled_map[i - 1][j] != self.filled_map[i + 1][j]:
                    self.filled_map[i][j] = self.door_place_tile

    def put_door_horizontal(self):
        start = None
        for i in range(self.height):
            for j in range(self.width):

                if self.filled_map[i][j] == self.walls_tile \
                        and self.in_map_field(i, j + 1) \
                        and self.filled_map[i][j + 1] == self.door_place_tile:
                    start = (i, j)
                    j += 1
                    while self.in_map_field(i, j) and self.filled_map[i][j] != self.walls_tile:
                        j += 1
                    end = (i, j)
                    if end[1] - start[1] >= 2:
                        self.doors_positions.append({
                            "coord": (start[0] + 1, start[1]),
                            "start": 2,
                            "end": 3,
                        })
                        self.filled_map[start[0]][start[1] + 1] = self.door_tile
                        self.filled_map[start[0]][start[1] + 2] = self.door_tile

    def put_door_vertical(self):
        start = None
        for j in range(self.width):
            for i in range(self.height):
                if self.filled_map[i][j] == self.walls_tile \
                        and self.in_map_field(i + 1, j) \
                        and self.filled_map[i + 1][j] == self.door_place_tile:
                    start = (i, j)
                    i += 1
                    while self.in_map_field(i, j) and self.filled_map[i][j] != self.walls_tile:
                        i += 1
                    end = (i, j)
                    if end[0] - start[0] >= 2:
                        self.doors_positions.append({
                            "coord": (start[0] + 1, start[1]),
                            "start": 2,
                            "end": 1,
                        })
                        self.filled_map[start[0] + 1][start[1]] = self.door_tile
                        self.filled_map[start[0] + 2][start[1]] = self.door_tile

    def find_all_doors_pos(self):
        self.create_filled_map()
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] not in self.used_tiles:
                    self.filled_map[i][j] = self.walls_tile
        self.find_all_door_pos_horizontal()
        self.find_all_door_pos_vertical()
        self.put_door_horizontal()
        self.put_door_vertical()
        for i in range(self.height):
            for j in range(self.width):
                if self.filled_map[i][j] == self.door_tile:
                    self.map[i][j] = self.filled_map[i][j]

    def dfs_components(self, nodes, idx, color):
        nodes[idx] = color
        for node in self.graph[idx]:
            if nodes[node] == -1:
                self.dfs_components(nodes, node, color)

    def find_components(self):
        nodes = [-1] * len(self.rooms)
        n_components = 0
        for i in range(len(self.rooms)):
            if nodes[i] == -1:
                n_components += 1
                self.dfs_components(nodes, i, n_components)
        self.components = [set() for i in range(n_components)]
        for i in range(len(self.rooms)):
            self.components[nodes[i] - 1].add(i)

    def put_entrance_door(self, idx):
        assert 0 <= idx < len(self.rooms)
        room = self.rooms[idx]
        for i in range(1, room._len - 2):
            if self.filled_map[room.y + i][room.x] == self.walls_tile \
                    and self.filled_map[room.y + i + 1][room.x] == self.walls_tile:
                self.filled_map[room.y + i][room.x] = self.door_tile
                self.filled_map[room.y + i + 1][room.x] = self.door_tile
                return True
            if self.filled_map[room.y + i][room.x + room.width - 1] == self.walls_tile \
                    and self.filled_map[room.y + i + 1][room.x + room.width - 1] == self.walls_tile:
                self.filled_map[room.y + i][room.x + room.width - 1] = self.door_tile
                self.filled_map[room.y + i + 1][room.x + room.width - 1] = self.door_tile
                return True
        for j in range(1, room.width - 2):
            if self.filled_map[room.y][room.x + j] == self.walls_tile \
                    and self.filled_map[room.y][room.x + j + 1] == self.walls_tile:
                self.filled_map[room.y][room.x + j] = self.door_tile
                self.filled_map[room.y][room.x + j + 1] = self.door_tile
                return True
            if self.filled_map[room.y + room._len - 1][room.x + j] == self.walls_tile \
                    and self.filled_map[room.y + room._len - 1][room.x + j + 1] == self.walls_tile:
                self.filled_map[room.y + room._len - 1][room.x + j] = self.door_tile
                self.filled_map[room.y + room._len - 1][room.x + j + 1] = self.door_tile
                return True
        return False

    def put_entrance_doors(self):
        if len(self.graph) == 0:
            self.prepare_building()
        for component in self.components:
            for node_idx in component:
                if self.put_entrance_door(node_idx):
                    break

    def update_doors_tiles_by_idx(self, idx):
        assert isinstance(self.rooms[idx], Room)
        room = self.rooms[idx]
        idx += 1
        for i in range(room._len):
            if i == 0 or i == room._len - 1:
                if i == 0:
                    for j in range(room.width):
                        if self.filled_map[room.y + i][room.x + j] == self.door_tile:
                            self.tiles_map_walls_doors[room.y + i][room.x + j] = self.top_horizontal_door_tile
                else:
                    for j in range(room.width):
                        if self.filled_map[room.y + i][room.x + j] == self.door_tile:
                            self.tiles_map_walls_doors[room.y + i][room.x + j] = self.bottom_horizontal_door_tile
            else:
                if self.filled_map[room.y + i][room.x] == self.door_tile:
                    self.tiles_map_walls_doors[room.y + i][room.x] = self.left_vertical_door_tile
                if self.filled_map[room.y + i][room.x + room.width - 1] == self.door_tile:
                    self.tiles_map_walls_doors[room.y + i][room.x + room.width - 1] = self.right_vertical_door_tile

    def update_all_doors_tiles(self):
        for i in range(len(self.rooms) - 1, -1, -1):
            self.update_doors_tiles_by_idx(i)

    def draw_room_by_idx(self, idx):
        assert isinstance(self.rooms[idx], Room)
        room = self.rooms[idx]
        idx += 1
        for i in range(room._len):
            if i == 0 or i == room._len - 1:
                if i == 0:
                    for j in range(room.width):
                        self.map[room.y + i][room.x + j] = idx
                        self.tiles_map_walls_doors[room.y + i][room.x + j] = self.top_horizontal_wall
                    self.tiles_map_walls_doors[room.y + i][room.x] = self.top_left_corner
                    self.tiles_map_walls_doors[room.y + i][room.x + room.width - 1] = self.top_right_corner
                else:
                    for j in range(room.width):
                        self.map[room.y + i][room.x + j] = idx
                        self.tiles_map_walls_doors[room.y + i][room.x + j] = self.bottom_horizontal_wall
                    self.tiles_map_walls_doors[room.y + i][room.x] = self.bottom_left_corner
                    self.tiles_map_walls_doors[room.y + i][room.x + room.width - 1] = self.bottom_right_corner
            else:
                self.map[room.y + i][room.x] = idx
                self.tiles_map_walls_doors[room.y + i][room.x] = self.left_vertical_wall
                self.map[room.y + i][room.x + room.width - 1] = idx
                self.tiles_map_walls_doors[room.y + i][room.x + room.width - 1] = self.right_vertical_wall

    def in_map_field(self, i, j):
        return i >= 0 and j >= 0 and i < self.height and j < self.width

    def horisontal_dfs(self, i, j, dy, dx, array, dy_list, dx_list):
        if self.map[i][j] not in self.used_tiles:
            updated_idx = None
            if dy + dx > 0:
                updated_idx = 1
            else:
                updated_idx = 0
            array[updated_idx][0] += dy
            array[updated_idx][1] += dx
            self.map[i][j] = self.walls_tile
            self.horisontal_dfs(i + dy, j + dx, dy, dx, array, dy_list, dx_list)
        else:
            i, j = i - dy, j - dx
            if self.tiles_map_walls_doors[i][j] == self.top_right_corner \
                    or self.tiles_map_walls_doors[i][j] == self.top_horizontal_wall:
                array.append("top")
            else:
                array.append("bottom")


    def vertical_dfs(self, i, j, dy, dx, array, dy_list, dx_list):
        if self.map[i][j] not in self.used_tiles:
            updated_idx = None
            if dy + dx > 0:
                updated_idx = 1
            else:
                updated_idx = 0
            array[updated_idx][0] += dy
            array[updated_idx][1] += dx
            self.map[i][j] = self.walls_tile
            self.vertical_dfs(i + dy, j + dx, dy, dx, array, dy_list, dx_list)
        else:
            i, j = i - dy, j - dx
            if self.tiles_map_walls_doors[i][j] == self.bottom_left_corner \
                    or self.tiles_map_walls_doors[i][j] == self.left_vertical_wall:
                array.append("left")
            else:
                array.append("right")

    def call_dfs(self, i, j, dy, dx, idx, array, dy_list, dx_list, function):
        saved_i, saved_j = i, j
        i, j = i + dy, j + dx
        if self.map[i][j] not in self.used_tiles:
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
                if self.map[i][j] not in self.used_tiles:
                    horizontal_idx, vertical_idx = None, None
                    for k in range(4):
                        if k < 2:
                            horizontal_idx = self.call_dfs(i, j,
                                                           dy[k],
                                                           dx[k],
                                                           horizontal_idx,
                                                           self.horizontal,
                                                           dy,
                                                           dx,
                                                           self.horisontal_dfs)
                        else:
                            if horizontal_idx is None:
                                vertical_idx = self.call_dfs(i, j,
                                                             dy[k],
                                                             dx[k],
                                                             vertical_idx,
                                                             self.vertical,
                                                             dy,
                                                             dx,
                                                             self.vertical_dfs)

    def fill_room_by_tile_num(self, idx, tile_num):
        room = self.rooms[idx]
        for i in range(1, room._len - 1):
            for j in range(1, room.width - 1):
                if self.map[room.y + i][room.x + j] > idx:
                    self.map[room.y + i][room.x + j] = -1
                    self.tiles_map_walls_doors[room.y + i][room.x + j] = self.zero_tile
                self.tiles_map_floor[room.y + i][room.x + j] = tile_num
        for j in range(room.width):
            self.tiles_map_floor[room.y][room.x + j] = tile_num
            self.tiles_map_floor[room.y + room._len - 1][room.x + j] = tile_num
        for i in range(room._len):
            self.tiles_map_floor[room.y + i][room.x] = tile_num
            self.tiles_map_floor[room.y + i][room.x + room.width - 1] = tile_num

    def create_map(self):
        self.map = [[self.outdoor_tile] * self.width for _ in range(self.height)]
        self.tiles_map_floor = [[self.outdoor] * self.width for _ in range(self.height)]
        self.tiles_map_walls_doors = [[self.zero_tile] * self.width for _ in range(self.height)]
        for i in range(len(self.rooms) - 1, -1, -1):
            self.draw_room_by_idx(i)
        for i in range(len(self.rooms)):
            self.fill_room_by_tile_num(i, self.floor_tile)

    def converted_layer_to_xml(self, array, name="layer"):
        starting = f'''<layer id="1" name="{name}" width="{self.width}" height="{self.height}" locked="1">
  <data encoding="csv">'''
        layer = '\n'.join(
            [','.join(map(str, array[i]))
             for i in range(self.height)])
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
            height = self.tile_size // 2
            if line[2] == 'bottom':
                y = line[0][0] * self.tile_size + self.tile_size // 2
            shapes.append(f'''  <object id="" x="{x}" y="{y}" width="{width}" height="{height}"/>''')
        for line in self.vertical:
            x = line[0][1] * self.tile_size
            y = line[0][0] * self.tile_size
            height = (line[1][0] - line[0][0] - 1) * self.tile_size
            width = self.tile_size // 2
            if line[2] == 'right':
                x = line[0][1] * self.tile_size + self.tile_size // 2
            shapes.append(f'''  <object id="" x="{x}" y="{y}" width="{width}" height="{height}"/>''')
        ending = ''' </objectgroup>'''
        return '\n'.join((starting, '\n'.join(shapes), ending))

    def convert_doors_to_xml(self):
        starting = '''<objectgroup id="8" name="Doors" locked="1">'''
        doors = []
        for door in self.doors_positions:
            x = door["coord"][1] * self.tile_size
            y = door["coord"][0] * self.tile_size
            start = door["start"]
            end = door["end"]
            doors.append(f'''  <object id="" x="{x}" y="{y}" start="{start}" end="{end}"/>''')
        ending = ''' </objectgroup>'''
        return '\n'.join((starting, '\n'.join(doors), ending))


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

    def precalc(self, building):
        building.create_map()#after that we have all layers with floor and walls
        #
        building.find_all_doors_pos()
        building.find_all_walls()#we can add types of walls in this method
        building.prepare_building()
        building.find_components()
        building.put_entrance_doors()
        building.update_all_doors_tiles()


def main():
    test_level = Level(n=5, rooms_count=5, max_coord=40, max_value=25, min_value=6)
    test_level.fit(20)
    i = 0
    building = test_level.population[i]
    test_level.precalc(building)
    # building.print(building.map)
    with open(f"generated_level_final{i}.tmx", 'w') as fout:
        building.create_image(f'level{i}.png')
        print(building.xml_initialization, file=fout)
        print(building.converted_layer_to_xml(building.tiles_map_floor,
                                                              name='floor'), file=fout)
        print(building.converted_layer_to_xml(building.tiles_map_walls_doors,
                                                              name='walls and doors'), file=fout)
        print(building.convert_shapes_to_xml(), file=fout)
        print(building.convert_doors_to_xml(), file=fout)
        print(building.xml_ending, file=fout)

    # test_level.population[0].prepare_building()
    # test_level.population[0].create_image("graph_implementing.png")
    # pprint(test_level.population[0].graph)
    # with open('almost_done.pkl', 'wb') as fout:
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

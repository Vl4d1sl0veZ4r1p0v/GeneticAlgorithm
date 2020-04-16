from pickle import load
import math
from PIL import Image, ImageDraw
import os
from GeneticAlgorithm import Building, Room, Level


class LevelContainer:

    def __init__(self, population):
        self.width = 70
        self.height = 60
        self.teil_size = 16
        self.population = population
        self.count_rooms = len(population[0].rooms)
        #need for building
        self.graph = []

    def create_image(self, filename, idx):
        img = Image.new("RGB", (self.width * self.teil_size, self.height * self.teil_size))
        img1 = ImageDraw.Draw(img)
        for room in self.population[idx].rooms:
            x, y, w, l = map(lambda x: x * self.teil_size,
                             (room.x, room.y, room.width, room._len))
            shape = [(x, y), (x + w, y + l)]
            img1.rectangle(shape, outline="#800080")
        path = os.getcwd() + '/' + filename
        img.save(path)


class ConvertXML:

    def __init__(self, out_tile=1, internal_tile=2, wall_tile=3):
        self.out_tile = out_tile
        self.internal_tile = internal_tile
        self.wall_tile = wall_tile

    def convert(self):
        pass


def main():
    level = None
    with open('offered_generated.pkl', 'rb') as fin:
        level = load(fin)
    test_container = LevelContainer(level)
    test_container.create_image("offered_generation2.png", 1)

if __name__ == "__main__":
    main()

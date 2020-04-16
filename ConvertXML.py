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


class ConvertXML:

    def __init__(self, out_tile=1, internal_tile=2, wall_tile=3):
        self.out_tile = out_tile
        self.internal_tile = internal_tile
        self.wall_tile = wall_tile

    def convert(self):
        pass


def main():
    pass
    # level = None
    # with open('offered_generated.pkl', 'rb') as fin:
    #     level = load(fin)
    # test_container = LevelContainer(level)
    # if __name__ == '__main__':

        # color = 128
        # image = Image.open("Background.png")
        #
        # # Draw some lines
        # draw = ImageDraw.Draw(image)
        # y_start = 0
        # y_end = image.height
        # step_size = 16
        #
        # for x in range(0, image.width, step_size):
        #     line = ((x, y_start), (x, y_end))
        #     draw.line(line, fill=color)
        #
        # x_start = 0
        # x_end = image.width
        #
        # for y in range(0, image.height, step_size):
        #     line = ((x_start, y), (x_end, y))
        #     draw.line(line, fill=color)
        #
        # del draw
        # image.show()


if __name__ == "__main__":
    main()

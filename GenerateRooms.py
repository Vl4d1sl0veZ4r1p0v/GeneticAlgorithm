#!/usr/bin/env python
# coding: utf-8

# Evaluation functions

# In[ ]:


from math import exp


# In[ ]:


def level_building():
    result = -1
    for chromosome in gene:
        if chromosome is first or chromosome is overlap:
            fil_corresponding_segment(chromosome)
    for adjacent_room in adjacent_rooms:
        place_door(adjacent_room)


# In[ ]:


#basic evaluations
def maximize_rooms():
    return len(rooms)


# In[ ]:


def maximize_total_rooms_area():
    return area(rooms)


# In[ ]:


def minimize_total_rooms_area():
    return 1000 * len(rooms) - area(rooms)


# In[ ]:


#graph evaluations
def maximize_degree():
    return get_sum_of_degrees()

def get_sum_of_degrees():
    return sum([degree(room) for room in rooms])


# In[ ]:


def maximize_diameter():
    return 1000 * len(rooms) + diameter(rooms)


# In[ ]:


def minimize_diameter():
    return 1000 * len(rooms) - diameter(rooms)


# In[ ]:


#corridor penalty
def ten_pow_tiny():
    return 10 ** count_of_tiny(rooms)

def corridor_penalty():
    return len(rooms) / ((1 + count_of_narrow(rooms)) * ten_pow_tiny(rooms))


# In[ ]:


def get_avg_degree():
    count = get_sum_of_degrees()
    return count / len(rooms)
        

def exp_degree():
    return exp((-1 * get_avg_degree - 2) ** 2)

def complex_fitness():
    return exp_degree() * len(rooms) * ln(diameter(rooms)) / 
(ln(e + 10 **  ten_pow_tiny()))


# Code of genetic algorithm

# Class room

# In[ ]:


class Room:
    
    def __init__(self, x, y, lenght, width, types):
        self.x = x
        self.y = y
        self.lenght = length
        self.width = width
        self.types = types
        
    def get_area(self):
        return self.length * self.width


# Class Building

# In[ ]:


class Building:
    
    def __init__(self, rooms: list):
        self.rooms = rooms
        
    def get_area():
        return 


# Class Level

# In[ ]:





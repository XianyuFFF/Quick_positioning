import numpy as np
import pickle

class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def __str__(self):
        return self.name + " : " + str(np.median(self.scores))



student1 = Student('ana', np.array([80,90,100,60]))
student2 = Student('ame', np.array([100, 98, 100, 40, 100, 100]))

students = [student1, student2]

with open('students', 'ab') as dbfile:
    pickle.dump(students, dbfile)


with open('students', 'rb') as dbfile:
    db = pickle.load(dbfile)

__author__ = 'thanh'

#################################################
# io
import os
rootpath = os.path.split(os.getcwd())[0]
trainpath = os.path.join(rootpath, "data\\train.json\\train.json")
testpath = os.path.join(rootpath, "data\\test.json\\test.json")
returnpath = os.path.join(rootpath, "return\\submission.csv")

#################################################
# extract info from each receipt
class ExtractRecipe():
    def __init__(self, recipe):
        # return information from each recipe
        # recipe: a complete recipe as dictionary
        self.id = self.getId(recipe)
        self.cuisine = self.getCuisine(recipe)
        self.ingredients = self.getIngredients(recipe)

    def getId(self, recipe):
        try:
            return recipe['id']
        except KeyError:
            return '-1'

    def getCuisine(self, recipe):
        try:
            return recipe['cuisine']
        except KeyError:
            return '-1'

    def getIngredients(self, recipe):
        try:
            return recipe['ingredients']
        except:
            return []

    def getData(self):
        return {
            'cuisine': self.cuisine,
            'ingredients': ', '.join([x for x in self.ingredients]),
            'id': self.id
        }

#################################################
# Loads JSON
def loadJS(path):
    import json
    from pandas import DataFrame
    from sklearn.preprocessing import LabelEncoder

    # each x is a object or dictionary in json, X is a dataframe
    data_frame = DataFrame([ExtractRecipe(x).getData() for x in json.load(open(path))])
    # encode labels with value between 0 and n_classes-1

    target_transform = LabelEncoder()
    data_frame['cuisine'] = target_transform.fit_transform(data_frame['cuisine'])  # fit encoder and return encoded labels

    return data_frame,target_transform

#################################################
# Write to predicted results to CSV
def write2CSV(file_path, rows):
    import csv
    with open(file_path, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['id', 'cuisine'])
        for row in rows:
            wr.writerow(row)

#################################################

from top2vec import Top2Vec

data = ""
with open('docs.txt') as file:
    data = file.read().replace('\n', '')

data = data.split(',')
print(len(data))


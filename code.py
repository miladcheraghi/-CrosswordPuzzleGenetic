import random
from copy import deepcopy
import pprint
import pandas
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def init():
    for k in range(0,50):
        for i in range(0,len(skeleton)):
            for j in range(0,len(skeleton[i])):
                if(skeleton[i][j] != 0):
                    skeleton[i][j] = letter[random.randint(0,31)]
        population.append(deepcopy(skeleton))

def fitness(population):
    costList = []
    for individual in population:
        cost = 0
        for i in range(0 , len(individual)):
            indices = [i for i, x in enumerate(individual[i]) if x == 0]
            if(len(indices) == 0):
                word = ''.join(individual[i][::-1])
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
            for k in range(0,len(indices)):
                if(k == 0):
                    temp = individual[i][:indices[k]]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
                if(k == len(indices)-1):
                    temp = individual[i][indices[k]+1:]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
                    continue
                temp = individual[i][indices[k]+1:indices[k+1]]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
        for i in range(0 , len(individual)): 
            column = [row[i] for row in individual]
            indices = [i for i, x in enumerate(column) if x == 0]
            if(len(indices) == 0):
                word = ''.join(column)
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
            for k in range(0,len(indices)):
                if(k == 0):
                    temp = column[:indices[k]]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
                if(k == len(indices)-1):
                    temp = column[indices[k]+1:]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
                    continue
                temp = column[indices[k]+1:indices[k+1]]
                word = ''.join(temp)
                if(len(word)>1):
                        if(word not in wordList):
                            cost += 1
        costList.append(cost)
    return costList

def crossover(population , costList , generationNum):
    childrenList = []
    sortedList = sorted(zip(costList,population))
    num = len(population) * 1/2
    if( num%2 != 0):
        num += 1
    num /= 2
    num = int(num)
    for i in range(0,num):
        mother = sortedList.pop(0)[1]
        father = sortedList.pop(0)[1]
        child1 = deepcopy(mother)
        child2 = deepcopy(father)
        rows = len(mother)
        # Scrolling rows
        for i in range(0,rows):
            # mother 
            motherIndices = [i for i, x in enumerate(mother[i]) if x == 0]
            if(len(motherIndices) == 0):
                word = ''.join(mother[i][::-1])
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(0,len(mother[i])):
                            child2[i][j] = mother[i][j]
            for k in range(0,len(motherIndices)):
                if(k == 0):
                    temp = mother[i][:motherIndices[k]]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(0 , motherIndices[k]):
                                child2[i][j] = mother[i][j]
                if(k == len(motherIndices)-1):
                    temp = mother[i][motherIndices[k]+1:]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(motherIndices[k]+1 , len(mother[i])):
                                child2[i][j] = mother[i][j]
                    continue
                temp = mother[i][motherIndices[k]+1:motherIndices[k+1]]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(motherIndices[k]+1 , motherIndices[k+1]):
                            child2[i][j] = mother[i][j]
            
            # father
            fatherIndices = [i for i, x in enumerate(father[i]) if x == 0]
            if(len(fatherIndices) == 0):
                word = ''.join(father[i][::-1])
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(0,len(father[i])):
                            child1[i][j] = father[i][j]
            for k in range(0,len(fatherIndices)):
                if(k == 0):
                    temp = father[i][:fatherIndices[k]]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(0 , fatherIndices[k]):
                                child1[i][j] = father[i][j]
                if(k == len(fatherIndices)-1):
                    temp = father[i][fatherIndices[k]+1:]
                    word = ''.join(temp[::-1])
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(fatherIndices[k]+1 , len(father[i])):
                                child1[i][j] = mother[i][j]
                    continue
                temp = father[i][fatherIndices[k]+1:fatherIndices[k+1]]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(fatherIndices[k]+1 , fatherIndices[k+1]):
                            child1[i][j] = father[i][j]
        columns = len(mother)
        # Scrolling columns
        for i in range(0,columns):
            # mother
            motherColumn = [row[i] for row in mother]
            motherIndices = [i for i, x in enumerate(motherColumn) if x == 0]
            if(len(motherIndices) == 0):
                word = ''.join(motherColumn)
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(0 , len(motherColumn)):
                            child2[j][i] = motherColumn[j]
            for k in range(0,len(motherIndices)):
                if(k == 0):
                    temp = motherColumn[:motherIndices[k]]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(0,motherIndices[k]):
                                child2[j][i] = motherColumn[j]
                if(k == len(motherIndices)-1):
                    temp = motherColumn[motherIndices[k]+1:]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(motherIndices[k]+1 , len(motherColumn)):
                                child2[j][i] = motherColumn[j]
                    continue
                temp = motherColumn[motherIndices[k]+1:motherIndices[k+1]]
                word = ''.join(temp)
                if(len(word)>1):
                        if(word in wordList):
                            for j in range(motherIndices[k]+1,motherIndices[k+1]):
                                child2[j][i] = motherColumn[j]

            # father
            fatherColumn = [row[i] for row in father]
            fatherIndices = [i for i, x in enumerate(fatherColumn) if x == 0]
            if(len(fatherIndices) == 0):
                word = ''.join(fatherColumn)
                if(len(word)>1):
                    if(word in wordList):
                        for j in range(0 , len(fatherColumn)):
                            child1[j][i] = fatherColumn[j]
            for k in range(0,len(fatherIndices)):
                if(k == 0):
                    temp = fatherColumn[:fatherIndices[k]]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(0,fatherIndices[k]):
                                child1[j][i] = fatherColumn[j]
                if(k == len(fatherIndices)-1):
                    temp = fatherColumn[fatherIndices[k]+1:]
                    word = ''.join(temp)
                    if(len(word)>1):
                        if(word in wordList):
                            for j in range(fatherIndices[k]+1 , len(fatherColumn)):
                                child1[j][i] = fatherColumn[j]
                    continue
                temp = fatherColumn[fatherIndices[k]+1:fatherIndices[k+1]]
                word = ''.join(temp)
                if(len(word)>1):
                        if(word in wordList):
                            for j in range(fatherIndices[k]+1,fatherIndices[k+1]):
                                child1[j][i] = fatherColumn[j]

        # mutation for child1
        if(random.uniform(0, 1) < 0.8):
            # print('mutation for child1')
            # print('befor')
            # pprint.pprint(child1)
            child1 = mutation(child1)
            if(generationNum > 50):
                child1 = mutation(child1)
                child1 = mutation(child1)
                child1 = mutation(child1)
                child1 = mutation(child1)
                child1 = mutation(child1)
            # print('after')
            # pprint.pprint(child1)
        # mutation for child2
        if(random.uniform(0, 1) < 0.8):
            # print('mutation for child2')
            # print('befor')
            # pprint.pprint(child2)
            child2 = mutation(child2)
            if(generationNum > 50):
                child2 = mutation(child2)
                child2 = mutation(child2)
                child2 = mutation(child2)
                child2 = mutation(child2)
                child2 = mutation(child2)
            # print('after')
            # pprint.pprint(child2)
        # input()

        childrenList.append(child1)
        childrenList.append(child2)        
    return childrenList

def mutation(children):
    rowOrColumn = random.randint(1, 2)
    number = random.randint(0, len(skeleton)-1)
    # print(rowOrColumn , number)
    # select word from row
    if(rowOrColumn == 1):
        childrenIndices = [i for i, x in enumerate(children[number]) if x == 0]
        if(len(childrenIndices) == 0):
            word = ''.join(children[number][::-1])
            if(len(word)>1):
                selectedWord = searchWord(word)
                selectedWord = selectedWord[::-1]
                for j in range(0,len(selectedWord)):
                    children[number][j] = selectedWord[j]
        for k in range(0,len(childrenIndices)):
            if(k == 0):
                temp = children[number][:childrenIndices[k]]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    selectedWord = searchWord(word)
                    selectedWord = selectedWord[::-1]
                    for j in range(0,childrenIndices[k]):
                        children[number][j] = selectedWord[j]
            if(k == len(childrenIndices)-1):
                temp = children[number][childrenIndices[k]+1:]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    selectedWord = searchWord(word)
                    selectedWord = selectedWord[::-1]
                    for j in range(childrenIndices[k]+1, len(children[number]) ):
                        children[number][j] = selectedWord[j - (childrenIndices[k]+1)]
                continue
            temp = children[number][childrenIndices[k]+1:childrenIndices[k+1]]
            word = ''.join(temp[::-1])
            if(len(word)>1):
                selectedWord = searchWord(word)
                selectedWord = selectedWord[::-1]
                for j in range(childrenIndices[k]+1, childrenIndices[k+1] ):
                    children[number][j] = selectedWord[j - (childrenIndices[k]+1)]
    # select word from column
    if(rowOrColumn == 2):
        column = [row[number] for row in children]
        childrenIndices = [i for i, x in enumerate(column) if x == 0]
        if(len(childrenIndices) == 0):
            word = ''.join(column)
            if(len(word)>1):
                selectedWord = searchWord(word)
                for j in range(0,len(selectedWord)):
                    children[j][number] = selectedWord[j]
        for k in range(0,len(childrenIndices)):
            if(k == 0):
                temp = column[:childrenIndices[k]]
                word = ''.join(temp)
                if(len(word)>1):
                    selectedWord = searchWord(word)
                    for j in range(0,childrenIndices[k]):
                        children[j][number] = selectedWord[j]
            if(k == len(childrenIndices)-1):
                temp = column[childrenIndices[k]+1:]
                word = ''.join(temp)
                if(len(word)>1):
                    selectedWord = searchWord(word)
                    for j in range(childrenIndices[k]+1,len(column)):
                        children[j][number] = selectedWord[j - (childrenIndices[k]+1)]
                continue
            temp = column[childrenIndices[k]+1:childrenIndices[k+1]]
            word = ''.join(temp)
            if(len(word)>1):
                selectedWord = searchWord(word)
                for j in range(childrenIndices[k]+1,childrenIndices[k+1]):
                    children[j][number] = selectedWord[j - (childrenIndices[k]+1)]
    
    return children

def searchWord(word):
    length = len(word)
    counter = random.randint(0, len(wordList))
    while( counter < len(wordList) ):
        if(length == len(wordList[counter])):
            word2 = wordList[counter]
            break
        counter += 1
        if(counter == len(wordList)-1):
            counter = 0 
    return word2

def selectSurvivor(population,costList):
    # sortedList = sorted(zip(costList,population))
    num = len(population) * 0.28
    for counter in range(0,int(num)):
        index = costList.index(max(costList))
        costList.remove(costList[index])
        population.remove(population[index])
    if(len(population) > 100):
        length = len(population)
        for i in range(length , 100 , -1):
            index = costList.index(max(costList))
            costList.remove(costList[index])
            population.remove(population[index])
    return population

def maxCost(skeleton):
    for i in range(0,len(skeleton)):
        for j in range(0,len(skeleton[i])):
            if(skeleton[i][j] != 0):
                skeleton[i][j] = 'ب'
    cost = 0
    for i in range(0 , len(skeleton)):
        indices = [i for i, x in enumerate(skeleton[i]) if x == 0]
        if(len(indices) == 0):
            word = ''.join(skeleton[i][::-1])
            if(len(word)>1):
                if(word not in wordList):
                    cost += 1
        for k in range(0,len(indices)):
            if(k == 0):
                temp = skeleton[i][:indices[k]]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
            if(k == len(indices)-1):
                temp = skeleton[i][indices[k]+1:]
                word = ''.join(temp[::-1])
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
                continue
            temp = skeleton[i][indices[k]+1:indices[k+1]]
            word = ''.join(temp[::-1])
            if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
    for i in range(0 , len(skeleton)): 
        column = [row[i] for row in skeleton]
        indices = [i for i, x in enumerate(column) if x == 0]
        if(len(indices) == 0):
            word = ''.join(column)
            if(len(word)>1):
                if(word not in wordList):
                    cost += 1
        for k in range(0,len(indices)):
            if(k == 0):
                temp = column[:indices[k]]
                word = ''.join(temp)
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
            if(k == len(indices)-1):
                temp = column[indices[k]+1:]
                word = ''.join(temp)
                if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
                continue
            temp = column[indices[k]+1:indices[k+1]]
            word = ''.join(temp)
            if(len(word)>1):
                    if(word not in wordList):
                        cost += 1
    return cost

def output(result):
    for i in range(0,len(result)):
        for j in range(0,len(result[i])):
            if( result[i][j] == 0 ):
                result[i][j] = '0'
    return result

if __name__ == '__main__':
    avgCost = np.empty(shape=[0,0])
    generationNum = np.empty(shape=[0,0])
    successfullWord = np.empty(shape=[0,0])
    # skeleton = [        
    #     [1,1,1,1],
    #     [0,1,0,1],
    #     [0,1,0,1],
    #     [1,1,1,1]
    # ]
    # skeleton = [        
    #     [1,1,1,1,0],
    #     [0,1,0,1,1],
    #     [0,1,0,1,1],
    #     [1,1,1,1,1],
    #     [1,1,1,1,1]
    # ]
    # american style
    # skeleton = [        
    #     [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
    #     [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
    #     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     [1,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
    #     [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0],
    #     [0,0,1,1,1,0,1,1,1,0,1,1,1,1,1],
    #     [1,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
    #     [1,1,1,0,1,1,1,1,1,1,1,0,1,1,1],
    #     [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],
    #     [1,1,1,1,1,0,1,1,1,0,1,1,1,0,0],
    #     [0,0,0,1,1,1,1,1,0,1,1,1,1,1,1],
    #     [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1],
    #     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
    #     [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
    # ]
    # jepanies style
    skeleton = [        
        [1,1,1,1,0,1,1,1,1],
        [1,1,1,0,1,1,1,1,0],
        [1,1,0,1,1,0,1,0,1],
        [1,0,1,1,1,1,0,1,1],
        [1,1,1,1,0,1,1,1,1],
        [1,1,0,1,1,1,1,0,1],
        [1,0,1,0,1,1,0,1,1],
        [0,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1,1,1],
    ]
    # uk style 
    # skeleton = [        
    #     [1,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
    #     [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],
    #     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],
    #     [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
    #     [0,1,0,1,0,1,0,1,0,1,0,1,0,0,1],
    #     [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
    #     [0,0,0,1,0,1,0,1,0,1,0,1,0,0,0],
    #     [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
    #     [1,0,0,1,0,1,0,1,0,1,0,1,0,1,0],
    #     [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1],
    #     [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0],
    #     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0],
    #     [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],
    # ]
    size = len(skeleton)
    data = pandas.read_csv('dic.csv' , names='f' , encoding="utf-8")  
    wordList = data.f.tolist()
    letter =['ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح'
            ,'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س','ش'
            ,'ص' , 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق'
            ,'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی']
    population = []
    init()
    
    maximumCost = maxCost(skeleton)
    for counter in range(0,90):
        if(counter != 0):
            population = selectSurvivor(population,costList)
        costList = fitness(population)
        childrenList = crossover(population , costList , counter)
        for item in childrenList:
            population.append(item)
        avg = sum(costList) / float(len(costList))
        minCost = min(costList)
        avgCost = np.append(avgCost , avg)
        generationNum = np.append(generationNum , counter)
        successfullWord = np.append(successfullWord , maximumCost - int(minCost))
        print('---------------------------')
        print('generation #' , counter)
        print('average of cost = ' , avg)
        print('min of cost = ' , minCost)
        print('size of population = ' , len(population))
        if(minCost == 0):
            print(':)))))))))))))))))))))))))))))))))))))))))))))))))')
            print('congratulation!!\n An answer has been found...')
            sortedList = sorted(zip(costList,population))
            pprint.pprint(sortedList.pop(0)[1])
            break
   
    fig, ax = plt.subplots()
    ax.plot(generationNum , avgCost)
    ax.set(xlabel='Generations', ylabel='Average Of Cost',
        title='Japanese Style')
    ax.grid()
    fig.savefig("test.png")
    plt.show()


    plt.style.use('seaborn-whitegrid')
    plt.figure(1)
    plt.subplot(111)
    plt.plot(generationNum, successfullWord, '-ok', color='blue')
    plt.title('Japanese Style')
    plt.xlabel('Generations')
    plt.ylabel('Successful Words')

    plt.show()

    sortedList = sorted(zip(costList,population))
    pprint.pprint(output(sortedList.pop(0)[1]))

























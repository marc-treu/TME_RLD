#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:13:41 2019

@author: Treü Marc & Jules Bonnard
"""

import matplotlib.pyplot as plt
import numpy as np


def format(file_name):
    """
        On va modifier les séparateur du fichier d'entréé des données. En effet, il est composé de deux séparateurs
    différent, ; et :, on va donc l'uniformisé.

    :param file_name: Le nom du fichier qui contient les données
    """
    with open(file_name, "r") as file:

        file = file.read()
        file = file.replace(':', ';')
        
    with open(file_name, "w") as f:
        f.write(file)
            

# Exercice 2 Baseline

def base_line_random(data):
    """
        A chaque itération, on choisit n'importe quel annonceur

    :param data: Les données sur les annonceurs
    :return: Le score
    """
    score_random = []
    for i in range(len(data)):
        score_random.append(np.random.choice(data[i, 6:]))
    return score_random, np.sum(score_random)/len(data)


def base_line_staticBest(data):
    """
        A chaque itération, on choisit l'annonceur avec le meilleur taux de clics cumulés

    :param data: Les données sur les annonceurs
    :return: Le score
    """
    liste = np.sum(data[:, 6:], axis=0)
    index = np.argmax(liste)

    print(index)
    score = []
    for i in range(len(data)):
        score.append(data[i, index + 6])
    return score, np.sum(score)/len(data)


def base_line_optimale(data):
    """
        A chaque itération, on choisit l'annonceur qui a le meilleur taux de clics à cette itération

    :param data: Les données sur les annonceurs
    :return: Le score optimal
    """
    score = []
    for i in range(len(data)):
        score.append(max(data[i, 6:]))
    return score, np.sum(score)/len(data)


# Exercice 3 - UCB

def ucb(data):
    """
    
    :param data: 
    :return: 
    """
    
    # Initialisation
    moyenne = []  # la moyenne de nos machine
    score = []
    for i in range(10):
        moyenne.append((data[i, i+6], 1))
        score.append(data[i, i+6])
    
    # Boucle pricipale
    for i in range(10, len(data)):
        index = calcule(moyenne, i)
        moyenne[index] = (np.add(moyenne[index][0], data[i, index+6]), moyenne[index][1] + 1)
        score.append(data[i, index+6])

    return score, np.sum(score) / len(data)


def calcule(moyenne, t):
    temp = [i[0]/i[1] + np.sqrt(((2*np.log(t)) / i[1])) for i in moyenne]
    return temp.index(max(temp))


# Exercice 4 - LinUCB

def linucb(data, alpha=0.2):

    # Initialisation
    score = []
    A = [np.identity(5) for _ in range(10)]
    b = [np.zeros((5, 1)) for _ in range(10)]

    # Boucle pricipale
    for i in range(len(data)):

        temp = []
        x_t = data[i, 1:6].reshape((5, 1))
        for j in range(10):
            inv_A = np.linalg.inv(A[j])
            teta = inv_A.dot(b[j])
            temp.append(teta.T.dot(x_t) + alpha * np.sqrt(x_t.T.dot(inv_A).dot(x_t)))

        index = temp.index(max(temp))

        A[index] += x_t.dot(x_t.T)

        res = x_t.dot(data[i, index + 6])
        b[index] += res

        score.append(data[i, index + 6])

    return score, np.sum(score) / len(data)


if __name__ == '__main__':
    
    # A run la premier fois du programme. Sinon on a deux type de séparateur dans notre fichiers CTR.txt
    format('CTR.txt')

    donnee = np.genfromtxt('CTR.txt', delimiter=';')  # On lit les données du fichiers
    print("Les données on bien été lu\n")

    """
    On rappel le format des données.
    <numero de l'article>:<représentation de l'article en 5 dimensions>:<taux de clics sur les publicités de 10 
    annonceurs> 
    
    Ex:
    [14.          0.18015101  0.05988999  0.05056417  0.70257318  0.65853586
      0.          0.09720626  0.          0.25355877  0.05642384  0.
      0.          0.10956013  0.09864809  0.19204752]
    """


    # Exercice 2 - Réalisation des Baselines

    # Stratégie Random
    score_random = base_line_random(donnee)

    # Stratégie StaticBest
    score_staticBest = base_line_staticBest(donnee)

    # Stratégie Optimale
    score_optimale = base_line_optimale(donnee)
    print("On obtient les 3 baselines suivante:\nRandom :", score_random[1], "\nStaticBest :", score_staticBest[1],
          "\nOptimal :", score_optimale[1])




    # Exercice 3 - UCB
    score_ucb = ucb(donnee)
    print()
    print("UCB :", score_ucb[1])



    # Exercice 4 - LinUCB
    score_linucb = linucb(donnee)
    print()
    print("LinUCB :", score_linucb[1])

    x = [i for i in range(5000)]

    plt.plot(x, np.cumsum(score_random[0]))
    plt.plot(x, np.cumsum(score_ucb[0]))
    plt.plot(x, np.cumsum(score_staticBest[0]))
    plt.plot(x, np.cumsum(score_linucb[0]))
    plt.plot(x, np.cumsum(score_optimale[0]))

    plt.legend(['Perfect', 'LinUCB', 'staticBest', 'UCB', 'Random'], loc='upper left')

    plt.show()

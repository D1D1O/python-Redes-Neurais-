# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:53:23 2019

@author: didio
"""

import numpy as np

def sigmoid (soma):
    return 1/ (1 + np.exp(-soma))
def sigmoidDerivada(sig):
    return sig * (1 - sig)
a = sigmoid(0.5)
b = sigmoidDerivada(a)

#b = sigmoid(-1.5)
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])
    
pesos1 = np.array([[-0.017],[-0.893],[0.148]])

pesos0 = 2*np.random.random((2,3)) -1
pesos1 = 2*np.random.random((3,1)) -1

epocas = 100000
momento = 1
taxaAprendizagem = 0.6

for j in range(epocas):
    camadaEntrada = entradas
    SomaSinapse0 = np.dot(camadaEntrada,pesos0)
    camadaOculta = sigmoid(SomaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    
    print("Erro: " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
    
    
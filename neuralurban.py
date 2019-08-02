"""
Feito para o projeto iOwlT em 01/08/2019

Alunos:
    
    Davi Moreno
    Gabriel Firmo
    Matheus Farias

Professores:
    
    Daniel Filgueiras
    Edna Natividade
"""
#Importações
import librosa
import numpy as np
from scipy.io import wavfile
import os
import soundfile
import re
import random
from shutil import copy, rmtree
from keras.layers import Dense
from scipy import signal
import keras.initializers
from keras.models import Sequential
import matplotlib.pyplot as plt
'''
    COLOCAR O UrbanSound8K NA MESMA PASTA QUE ESSE CODIGO
    PATHI -- Caminho da pasta que tem os folds
    PATH -- Caminho dos arquivos de todos os arquivos de tiro
    PATH2 -- Caminho para nova pasta que será criada com arquivos de tiro convertidos
    PATH3 -- Caminho para nova pasta que será criada com arquivos de não tiro janelados
    PATH4 -- Caminho para nova pasta que será criada com arquivos de tiro janelados
    AUX -- Pasta auxiliar onde ficará os sons que não sao tiros (8000+ dados) editados para 16 bits e 8k samples
    NEW_RATE -- Nova taxa de amostragem (samplerate) dos dados
    NEW_BIT_RATE -- Nova bitrate dos dados
    NEW_DIR -- Nome da pasta de tiros (gunshot_sounds é o padrão)
    PATTERN -- É o label de categoria do tiro no UrbanSounds (6)
    TAM_JANELA -- É o tamanho da janela que será analisada no som em segundos (recomendado: 0.4)
    ORDEM_FILTRO -- É a ordem do filtro passa banda utilizado para diminuir a possibilidade de aparecer ruidos (recomendado: 1)
    FCI -- É a frequência de corte inferior do filtro passa banda em Hz (recomendado: 100)
    FCS -- É a frequência de corte superior do filtro passa banda em Hz (recomendado: 2000)
''' 
PATHI = 'UrbanSound8K/audio/'
PATH = 'UrbanSound8K/audio/gunshot_sounds/'
PATH2 = 'UrbanSound8K/audio/tiros_convertidos/'
PATH3 = 'UrbanSound8K/audio/não_tiros_janelados/'
PATH4 = 'UrbanSound8K/audio/tiros_janelados/'
AUX = 'UrbanSound8K/audio/aux/'
NEW_RATE = 8000
NEW_BIT_RATE = 'PCM_16'
NEW_DIR = 'gunshot_sounds'
PATTERN = '6'
TAM_JANELA = 0.4
ORDEM_FILTRO = 1
FCI = 100
FCS = 2000

#Criando a pasta onde ficarão apenas os sons de tiro do UrbanSounds
PATH_NEW_DIR = PATHI + NEW_DIR
if not os.path.exists(PATH_NEW_DIR):
    os.mkdir(PATH_NEW_DIR)
label = []
audio=[]
for directory in os.listdir(PATHI):
    if 'fold' in directory:
        for file in os.listdir(PATHI + directory):
            if re.match("\d*\-"+PATTERN+"\-\d*\-",file):
                copy(PATHI + directory + "/" + file, PATH_NEW_DIR)

#Convertendo(samplerate e bitrate) e salvando(no PATH2) os arquivos .wav 
if not os.path.exists(PATH2):
    os.mkdir(PATH2)
for wav_file in os.listdir(PATH):
    data, rate = librosa.load(PATH + wav_file, sr=NEW_RATE)
    soundfile.write(PATH2 + wav_file, data, rate, subtype=NEW_BIT_RATE)                

#Cálculo do valor rms e desvio padrão do conjunto de treinamento para tiros
N=0
aux_data=0
for wav_file in os.listdir(PATH2):
    rate, data = wavfile.read(PATH2 + wav_file)
    aux_data = np.hstack( (data, aux_data) )
aux_data = aux_data/(2**15) 
rms = np.sqrt(np.mean(aux_data**2))
std = np.std(aux_data)

#Treshold estimado através do valor rms e desvio padrão
th = rms + std

samples = int(TAM_JANELA * NEW_RATE) #Número de amostras por janela

#Pegando arquivos de nao tiros janelados e adicionando ao PATH3

if not os.path.exists(PATH3):
    os.mkdir(PATH3)
if not os.path.exists(AUX):
    os.mkdir(AUX)
for directory in os.listdir(PATHI):
    print(directory)
    if 'fold' in directory:
        for wav_file in os.listdir(PATHI+directory):
            if re.match("\d*\-[^6]\-\d*\-",wav_file):
                data, rate = librosa.load(PATHI+ directory + "/" + wav_file, sr=NEW_RATE)
                soundfile.write(AUX+wav_file, data, rate, subtype=NEW_BIT_RATE)
                rate, data = wavfile.read(AUX + wav_file)
                data = data/2**15 #normalização
                for i in range(len(data)-10):
                    mean = np.mean( abs(data[i:i+10]) )
                    if mean >= 0.3*th:
                        if i + samples <= len(data):
                            window = data[i:(i+samples)]
                            soundfile.write(PATH3+wav_file, window, rate)
                            break


#Selecionando janela dos sons do dataset de treinamento a partir do treshold obtido
if not os.path.exists(PATH4):
    os.mkdir(PATH4)

for wav_file in os.listdir(PATH2):
    rate, data = wavfile.read(PATH2 + wav_file)
    data = data/2**15 #normalização
    for i in range(len(data)-10):
        mean = np.mean( abs(data[i:i+10]) )
        if mean >= 0.5*th:
            if i + samples <= len(data):
                window = data[i:(i+samples)]
                soundfile.write(PATH4 + wav_file, window, rate)
                break
            
            

#Leitura dos dados
PATHa = 'UrbanSound8K/audio/treino1/'
PATHa2 = 'UrbanSound8K/audio/treino0/'
PATHa3 = 'UrbanSound8K/audio/teste0/'
PATHa4 ='UrbanSound8K/audio/teste1/'

if os.path.exists(PATHa):
    rmtree(PATHa)
    os.mkdir(PATHa)
else:
    os.mkdir(PATHa)

if os.path.exists(PATHa2):
    rmtree(PATHa2)
    os.mkdir(PATHa2)
else:
    os.mkdir(PATHa2)

if os.path.exists(PATHa3):
    rmtree(PATHa3)
    os.mkdir(PATHa3)
else:
    os.mkdir(PATHa3)

if os.path.exists(PATHa4):
    rmtree(PATHa4)
    os.mkdir(PATHa4)
else:
    os.mkdir(PATHa4)
    
    

tirosjanelados = os.listdir(PATH4)
naojanelados = os.listdir(PATH3)
randomize = random.SystemRandom()
randomize.shuffle(tirosjanelados)
randomize = random.SystemRandom()
randomize.shuffle(naojanelados)

for i in range (0,249):
    copy(PATH4+tirosjanelados[i], PATHa+tirosjanelados[i])

for i in range (249,len(tirosjanelados)):
    copy(PATH4+tirosjanelados[i], PATHa4+tirosjanelados[i])

for i in range(0,249):
    copy(PATH3+naojanelados[i], PATHa2+naojanelados[i])    

for i in range(249,len(tirosjanelados)):
    copy(PATH3+naojanelados[i], PATHa3+naojanelados[i])

#Filtro passa banda
b,a = signal.butter(N=ORDEM_FILTRO, Wn=[FCI,FCS], btype="bandpass", analog=False, fs=NEW_RATE)

#Aplicando normalização e passando pelo filtro passa banda
dt=[]
lb=[]
for wav_file in os.listdir(PATHa):
    rate, data = wavfile.read(PATHa + wav_file)
    data = data/2**15 #normalização
    data = signal.filtfilt(b, a, data)
    dt.append(abs(data))
    lb.append(1)

for wav_file in os.listdir(PATHa2):
    rate, data = wavfile.read(PATHa2 + wav_file)
    data = data/2**15 #normalização
    data = signal.filtfilt(b, a, data)
    dt.append(abs(data))
    lb.append(0)
   
dt_test=[]
lb_test=[]

for wav_file in os.listdir(PATHa4):
    rate, data = wavfile.read(PATHa4 + wav_file)
    data = data/2**15 #normalização
    data = signal.filtfilt(b, a, data)
    dt_test.append(abs(data))
    lb_test.append(1)

for wav_file in os.listdir(PATHa3):
    rate, data = wavfile.read(PATHa3 + wav_file)
    data = data/2**15 #normalização
    data = signal.filtfilt(b, a, data)
    dt_test.append(abs(data))
    lb_test.append(0)

#Definindo os vetores de treino e teste
x_train = np.row_stack(dt)
y_train = np.row_stack(lb)
x_test = np.row_stack(dt_test)
y_test = np.row_stack(lb_test)

#Definindo a rede neural
model = None
model = Sequential()
INI = np.sqrt(2/3200)
model.add(Dense(100, input_dim=3200, activation='tanh', 
                kernel_initializer=keras.initializers.Constant(value=INI), 
                bias_initializer='zeros'))
model.add(Dense(10, activation='tanh', 
                kernel_initializer=keras.initializers.Constant(value=INI), 
                bias_initializer='zeros'))
INI = np.sqrt(1/10)
model.add(Dense(1, activation='sigmoid', 
                kernel_initializer=keras.initializers.Constant(value=INI), 
                bias_initializer='zeros'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Mensurando acuracia da rede antes de treina-la
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy Teste Antes: %.2f' % (accuracy*100))

#Treinamento da rede neural
history = model.fit(x_train, y_train,
          epochs=15,
          batch_size=32,validation_data=(x_test, y_test))

#Mensurando acuracia da rede depois de treina-la
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy Teste: %.2f' % (accuracy*100))

_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy Treinamento: %.2f' % (accuracy*100))

# Plotando acuracia de treino e validação
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plotando valor de custo do treino e validação
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Matriz confusão
#Linha 1 - Acerto não tiro - Erro não tiro
#Linha 2 - Erro tiro - Acerto tiro

conf_matrix=np.array([[0,0],[0,0]])
predict = model.predict(x_test)
for i in range(len(x_test)):
    if predict[i] > 0.5:
        if y_test[i]==1:
            conf_matrix[1,1]+=1 
        else: 
            conf_matrix[0,1]+=1
    else:
        if y_test[i]==0:
            conf_matrix[0,0]+=1 
        else:
            conf_matrix[1,0]+=1
    
print(conf_matrix)
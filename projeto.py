import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def carregar_imagens_opencv(diretorio_base, tamanho=(64, 64)):
    imagens = []
    labels = []
    
    for classe in os.listdir(diretorio_base):
        caminho_classe = os.path.join(diretorio_base, classe)
        
        if os.path.isdir(caminho_classe):
            for imagem_nome in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, imagem_nome)
                try:
                    imagem = cv2.imread(caminho_imagem)
                    imagem = cv2.resize(imagem, tamanho)
                    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                    imagem_normalizada = imagem / 255.0
                    imagens.append(imagem_normalizada)
                    labels.append(classe)
                except Exception as e:
                    print(f"Erro ao carregar imagem {caminho_imagem}: {e}")
    
    imagens = np.array(imagens)
    labels = np.array(labels)
    return imagens, labels

def construir_cnn(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Definir os caminhos
train_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/train'
valid_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/valid'
test_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/test'

# Carregar os dados
X_train, y_train = carregar_imagens_opencv(train_dir)
X_valid, y_valid = carregar_imagens_opencv(valid_dir)
X_test, y_test = carregar_imagens_opencv(test_dir)

# Codificar as labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_valid_encoded = encoder.transform(y_valid)
y_test_encoded = encoder.transform(y_test)

# Converter para one-hot encoding
y_train_cat = to_categorical(y_train_encoded, num_classes=len(np.unique(y_train)))
y_valid_cat = to_categorical(y_valid_encoded, num_classes=len(np.unique(y_valid)))
y_test_cat = to_categorical(y_test_encoded, num_classes=len(np.unique(y_test)))

# Definir parâmetros do modelo
input_shape = (64, 64, 3)
num_classes = len(np.unique(y_train))

# Construir e treinar a CNN
cnn_model = construir_cnn(input_shape, num_classes)
cnn_model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=10, batch_size=32)

# Avaliar o modelo
test_loss, test_acc = cnn_model.evaluate(X_test, y_test_cat)
print(f'Acurácia no teste: {test_acc}')

# Função para exibir imagens classificadas
def mostrar_exemplos(X_test, y_test_true, y_test_pred, num_exemplos=5):
    acertos = np.where(y_test_true == y_test_pred)[0]  # Índices dos acertos
    erros = np.where(y_test_true != y_test_pred)[0]    # Índices dos erros
    
    print(f"Mostrando {num_exemplos} exemplos de acertos:")
    for i in range(min(num_exemplos, len(acertos))):
        idx = acertos[i]
        plt.imshow(X_test[idx])
        plt.title(f"Verdadeiro: {encoder.inverse_transform([y_test_true[idx]])[0]}, Previsto: {encoder.inverse_transform([y_test_pred[idx]])[0]}")
        plt.show()
    
    print(f"Mostrando {num_exemplos} exemplos de erros:")
    for i in range(min(num_exemplos, len(erros))):
        idx = erros[i]
        plt.imshow(X_test[idx])
        plt.title(f"Verdadeiro: {encoder.inverse_transform([y_test_true[idx]])[0]}, Previsto: {encoder.inverse_transform([y_test_pred[idx]])[0]}")
        plt.show()

# Fazer previsões no conjunto de teste
y_test_pred_cat = cnn_model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_cat, axis=1)

# Mostrar exemplos de acertos e erros
mostrar_exemplos(X_test, y_test_encoded, y_test_pred, num_exemplos=5)

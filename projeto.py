import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Função para carregar imagens sem redimensionar ou com maior resolução (por exemplo, 128x128)
def carregar_imagens_opencv(diretorio_base, tamanho=(64, 64)):  # Pode alterar para (256, 256) ou manter resolução original
    imagens = []
    labels = []
    
    for classe in os.listdir(diretorio_base):
        caminho_classe = os.path.join(diretorio_base, classe)
        
        if os.path.isdir(caminho_classe):
            for imagem_nome in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, imagem_nome)
                try:
                    imagem = cv2.imread(caminho_imagem)
                    # Se quiser redimensionar, mantenha o código abaixo
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

# Função para construir a CNN
def construir_cnn(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Use GlobalAveragePooling2D para permitir que o modelo lide com resoluções maiores
    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Definir os caminhos
train_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/train'
valid_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/valid'
test_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/test'

# Carregar os dados (pode ajustar o tamanho para resolução desejada)
X_train, y_train = carregar_imagens_opencv(train_dir, tamanho=(64, 64))
X_valid, y_valid = carregar_imagens_opencv(valid_dir, tamanho=(64, 64))
X_test, y_test = carregar_imagens_opencv(test_dir, tamanho=(64, 64))

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
input_shape = (64, 64, 3)  # Atualizar para o novo tamanho da imagem (exemplo: 128x128x3)
num_classes = len(np.unique(y_train))

# Construir a CNN
cnn_model = construir_cnn(input_shape, num_classes)

# Aumento de dados - definindo transformações
datagen = ImageDataGenerator(
    rotation_range=20,      # Rotação de até 20 graus
    width_shift_range=0.2,  # Translação horizontal de até 20%
    height_shift_range=0.2, # Translação vertical de até 20%
    zoom_range=0.2,         # Zoom
    horizontal_flip=True,   # Espelhamento horizontal
    fill_mode='nearest'     # Preenchimento dos pixels fora da borda
)

# Treinamento com aumento de dados
cnn_model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32), 
    validation_data=(X_valid, y_valid_cat), 
    epochs=30
)

# Avaliar o modelo
test_loss, test_acc = cnn_model.evaluate(X_test, y_test_cat)
print(f'Acurácia no teste: {test_acc}')

# Fazer previsões no conjunto de teste
y_test_pred_cat = cnn_model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_cat, axis=1)

# Contar acertos e erros
acertos = np.sum(y_test_encoded == y_test_pred)
erros = np.sum(y_test_encoded != y_test_pred)

print(f'Número de acertos: {acertos}')
print(f'Número de erros: {erros}')

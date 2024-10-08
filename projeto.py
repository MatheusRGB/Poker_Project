import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Função para carregar imagens em escala de cinza
def carregar_imagens_opencv(diretorio_base, tamanho=(64, 64)):
    imagens = []
    labels = []
    
    for classe in os.listdir(diretorio_base):
        caminho_classe = os.path.join(diretorio_base, classe)
        
        if os.path.isdir(caminho_classe):
            for imagem_nome in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, imagem_nome)
                try:
                    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza
                    imagem = cv2.resize(imagem, tamanho)  # Redimensionar se necessário
                    imagem_normalizada = imagem / 255.0  # Normalizar os valores dos pixels
                    imagens.append(imagem_normalizada)
                    labels.append(classe)
                except Exception as e:
                    print(f"Erro ao carregar imagem {caminho_imagem}: {e}")
    
    imagens = np.array(imagens)
    imagens = np.expand_dims(imagens, axis=-1)  # Adicionar uma dimensão extra para o canal da imagem
    labels = np.array(labels)
    return imagens, labels

# Função para construir a CNN adaptada para escala de cinza
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

# Função para gerar imagens aumentadas
def gerar_imagens_aumentadas(datagen, X, y, num_augmentations):
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        img = X[i].reshape((1,) + X[i].shape)
        
        # Gerar várias imagens aumentadas
        for _ in range(num_augmentations):
            aug_iter = datagen.flow(img, batch_size=1)
            img_augmented = next(aug_iter)[0]  # Pega a primeira imagem gerada
            X_augmented.append(img_augmented)
            y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)

# Definir os caminhos
train_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/train'
valid_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/valid'
test_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/test'

# Carregar os dados em escala de cinza
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
input_shape = (64, 64, 1)  # Atualizar para 1 canal de cinza
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

# Gerar imagens aumentadas
X_train_augmented, y_train_augmented = gerar_imagens_aumentadas(datagen, X_train, y_train_cat, num_augmentations=5)

# Treinamento com menos épocas já que os dados foram aumentados antes
cnn_model.fit(
    X_train_augmented, y_train_augmented, 
    validation_data=(X_valid, y_valid_cat), 
    epochs=25  # Menos épocas pois a base já está aumentada
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

import os
import cv2
import numpy as np
import pickle  # Para salvar e carregar o encoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def carregar_imagens_opencv(diretorio_base, tamanho=(64, 64)):
    imagens = []
    labels = []
    
    for classe in os.listdir(diretorio_base):
        caminho_classe = os.path.join(diretorio_base, classe)
        
        if os.path.isdir(caminho_classe):
            for imagem_nome in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, imagem_nome)
                try:
                    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  
                    imagem = cv2.resize(imagem, tamanho)  
                    imagem_normalizada = imagem / 255.0  
                    imagens.append(imagem_normalizada)
                    labels.append(classe)
                except Exception as e:
                    print(f"Erro ao carregar imagem {caminho_imagem}: {e}")
    
    imagens = np.array(imagens)
    imagens = np.expand_dims(imagens, axis=-1)  
    labels = np.array(labels)
    return imagens, labels

def construir_cnn(input_shape, num_classes):
    model = Sequential()
    
    # Primeira camada
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Segunda camada
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Terceira camada
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Otimização para imagens maiores
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Camada Final
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

train_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/train'
valid_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/valid'
test_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/test'
model_path = 'modelo_cnn_cartas.h5' 
encoder_path = 'label_encoder.pkl'   

treinar_modelo = input("Deseja treinar o modelo? (s/n): ").strip().lower()

if treinar_modelo == 's' or not os.path.exists(model_path):
    X_train, y_train = carregar_imagens_opencv(train_dir, tamanho=(64, 64))
    X_valid, y_valid = carregar_imagens_opencv(valid_dir, tamanho=(64, 64))
    X_test, y_test = carregar_imagens_opencv(test_dir, tamanho=(64, 64))

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_valid_encoded = encoder.transform(y_valid)
    y_test_encoded = encoder.transform(y_test)

    # Salvar o encoder
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    y_train_cat = to_categorical(y_train_encoded, num_classes=len(np.unique(y_train)))
    y_valid_cat = to_categorical(y_valid_encoded, num_classes=len(np.unique(y_valid)))
    y_test_cat = to_categorical(y_test_encoded, num_classes=len(np.unique(y_test)))

    input_shape = (64, 64, 1)  
    num_classes = len(np.unique(y_train))

    cnn_model = construir_cnn(input_shape, num_classes)

    datagen = ImageDataGenerator(
        rotation_range=20,      
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        zoom_range=0.2,         
        horizontal_flip=True,   
        fill_mode='nearest'     
    )

    # Treinamento
    cnn_model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=32), 
        validation_data=(X_valid, y_valid_cat), 
        epochs=128
    )
    
    # Avaliar o modelo
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test_cat)
    print(f'Acurácia no teste: {test_acc}')

    # Salvar o modelo treinado
    cnn_model.save(model_path)
    print("Modelo treinado e salvo com sucesso.")
else:
    # Carregar modelo salvo
    cnn_model = load_model(model_path)
    print("Modelo carregado com sucesso.")

    # Carregar o encoder salvo
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder carregado com sucesso.")


# Classificar Input
def classificar_imagens_novas(diretorio, modelo, encoder):
    imagens = []
    nomes_imagens = []
    
    for imagem_nome in os.listdir(diretorio):
        caminho_imagem = os.path.join(diretorio, imagem_nome)
        
        if os.path.isfile(caminho_imagem):
            try:
                imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
                imagem = cv2.resize(imagem, (64, 64)) 
                imagem_normalizada = imagem / 255.0
                imagens.append(imagem_normalizada)
                nomes_imagens.append(imagem_nome)  
            except Exception as e:
                print(f"Erro ao carregar imagem {caminho_imagem}: {e}")
    
    if len(imagens) == 0:
        print("Nenhuma imagem encontrada.")
        return
    
    imagens = np.array(imagens)
    imagens = np.expand_dims(imagens, axis=-1)

    previsoes = modelo.predict(imagens)
    classes_previstas = encoder.inverse_transform(np.argmax(previsoes, axis=1))
    
    cartas_formatadas = ", ".join(classes_previstas)
    print(f"[{cartas_formatadas}]")

# Verifica Input
input_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/input1'
classificar_imagens_novas(input_dir, cnn_model, encoder)


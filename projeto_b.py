import os
import cv2
import numpy as np
import pickle  # Para salvar e carregar o encoder
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

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
model_path = 'C:/Users/Matheus Yago/Desktop/poker/archive/modelo_cnn_cartas.keras' 
encoder_path = 'C:/Users/Matheus Yago/Desktop/poker/archive/label_encoder.pkl'   

os.system('cls')
print("""
      
/$$$$$$$$           /$$                           /$$$$$$$                     
| $$__  $$         | $$                          | $$__  $$                    
| $$  \ $$ /$$$$$$ | $$   /$$  /$$$$$$   /$$$$$$ | $$  \ $$  /$$$$$$   /$$$$$$ 
| $$$$$$$//$$__  $$| $$  /$$/ /$$__  $$ /$$__  $$| $$$$$$$  /$$__  $$ /$$__  $$
| $$____/| $$  \ $$| $$$$$$/ | $$$$$$$$| $$  \__/| $$__  $$| $$  \__/| $$  \ $$
| $$     | $$  | $$| $$_  $$ | $$_____/| $$      | $$  \ $$| $$      | $$  | $$
| $$     |  $$$$$$/| $$ \  $$|  $$$$$$$| $$      | $$$$$$$/| $$      |  $$$$$$/
|__/      \______/ |__/  \__/ \_______/|__/      |_______/ |__/       \______/ 
                                                                                                                                                             
""")
print("                          |1 - Inicio Rapido| \n")
print("                       |2 - Iniciar com treinamento| \n")
treinar_modelo = input("Escolha como começar:\n").strip().lower()

if treinar_modelo == '2' or not os.path.exists(model_path):
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
        epochs=256
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
def classificar_input(diretorio, modelo, encoder):
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
    return(cartas_formatadas)
    #print(f"[{cartas_formatadas}]")

def classify_poker_hand(cards):
    values = [card.split(' ')[0] for card in cards]
    suits = [card.split(' ')[-1] for card in cards]
    
    value_counts = Counter(values)
    suit_counts = Counter(suits)
    
    value_count_list = sorted(value_counts.values(), reverse=True)
    
    is_flush = len(suit_counts) == 1
    
    rank_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'jack': 11, 'queen': 12, 'king': 13, 'ace': 14}
    ranks = sorted([rank_map[value] for value in values])

    is_straight = all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1))

    if is_flush and is_straight:
        return "Straight Flush"
    elif value_count_list == [4, 1]:
        return "Four of a Kind"
    elif value_count_list == [3, 2]:
        return "Full House"
    elif is_flush:
        return "Flush"
    elif is_straight:
        return "Straight"
    elif value_count_list == [3, 1, 1]:
        return "Three of a Kind"
    elif value_count_list == [2, 2, 1]:
        return "Two Pair"
    elif value_count_list == [2, 1, 1, 1]:
        return "One Pair"
    else:
        return "High Card"
    
def formatar_cartas(cartas_str):
    cartas_formatadas = [carta.strip() for carta in cartas_str.split(",")]
    return cartas_formatadas

def hand_score(hand):
    if hand == "Straight Flush":
        return 9
    if hand == "Four of a Kind":
        return 8
    if hand == "Full House":
        return 7
    if hand == "Flush":
        return 6
    if hand == "Straight":
        return 5
    if hand == "Three of a Kind":
        return 4
    if hand == "Two Pair":
        return 3
    if hand == "One Pair":
        return 2
    if hand == "High Card":
        return 1

# Carregar cartas
input_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/input1'
table_dir = 'C:/Users/Matheus Yago/Desktop/poker/archive/table'

hand = classificar_input(input_dir, cnn_model, encoder)
table = classificar_input(table_dir,cnn_model, encoder)
cards = hand + ", " + table

hand = formatar_cartas(hand)
table = formatar_cartas(table)
cards = formatar_cartas(cards) 
score = hand_score(classify_poker_hand(cards))

os.system('cls')
os.system('cls')
print("====================== POKERBRO ======================\n")

print("=================== HAND ===================\n")
print(hand)

print("\n=================== TABLE ==================\n")
print(table)

print("\n=================== SCORE ==================\n")

print("Você possui um " + classify_poker_hand(cards) + " - " + str(score) + " Pontos")

print("\n=================== CARDS ==================\n")
print(cards)

print("\n====================== POKERBRO ====================== \n")

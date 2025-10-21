import tensorflow as tf
from tensorflow import keras
from keras import layers
import os 

# --- USTAWIENIA ---
# Zakładam, że wróciliśmy do standardowego rozmiaru
IMAGE_SIZE = (400, 400) 
BATCH_SIZE = 32 

# Dynamiczne budowanie ścieżek
base_dir = os.path.dirname(os.path.realpath(__file__))

# Upewnij się, że te ścieżki wskazują na foldery stworzone przez 'przygotuj_letterbox.py'
TRAIN_DIR = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "train_letterboxed")
TEST_DIR = os.path.join(base_dir, "OpenAnimalTracks", "cropped_imgs", "test_letterboxed")
MODEL_FILENAME = os.path.join(base_dir, "moj_model_tropow_dostrojony.keras")

print(f"Ścieżka treningowa: {TRAIN_DIR}")
print(f"Ścieżka testowa: {TEST_DIR}")
print(f"Plik modelu: {MODEL_FILENAME}")

# Liczba epok
# Zwiększamy epoki, bo walka z przeuczeniem wymaga dłuższego treningu
INITIAL_EPOCHS = 200 
FINE_TUNE_EPOCHS = 100 # Możesz tu dać więcej, np. 50 
FINE_TUNE_LEARNING_RATE = 1e-5 # 0.00001

# --- KROK 1: WCZYTYWANIE I PRZYGOTOWANIE DANYCH TRENINGOWYCH ---

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"Znaleziono {NUM_CLASSES} klas: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# --- KROK 2: SPRAWDZANIE, CZY MODEL ISTNIEJE ---

# Zakładamy, że model został usunięty, więc wykona się sekcja 'else'
if os.path.exists(MODEL_FILENAME):
    print(f"Znaleziono zapisany model '{MODEL_FILENAME}'. Wczytywanie...")
    model = keras.models.load_model(MODEL_FILENAME)
    print("Model wczytany. Przechodzenie bezpośrednio do dostrajania (Fine-Tuning).")
    base_model = model.layers[0] # Uważaj, teraz pierwszym layerem będzie augmentacja!

else:
    # --- SCENARIUSZ B: MODEL NIE ISTNIEJE ---
    print(f"Nie znaleziono modelu '{MODEL_FILENAME}'. Tworzenie nowego...")
    
    # --- NOWA CZĘŚĆ: AUGMENTACJA "W LOCIE" ---
    # Definiujemy subtelne i realistyczne modyfikacje
    data_augmentation = keras.Sequential(
        [
            # Losowe odbicie lustrzane (lewo/prawo)
            layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            # Losowy obrót, ale tylko o 10% (ok. 36 stopni)
            layers.RandomRotation(0.1), 
            # Losowy zoom, ale tylko o 10%
            layers.RandomZoom(0.1),
            # Można też dodać małą zmianę kontrastu
            # layers.RandomContrast(0.1),
        ],
        name="augmentacja_w_locie"
    )
    # --- KONIEC NOWEJ CZĘŚCI ---

    # Krok 2.1: Budowa modelu bazowego
    base_model = tf.keras.applications.MobileNetV2(
        # Upewnij się, że rozmiar jest ten sam co IMAGE_SIZE
        input_shape=IMAGE_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False # Zamrażamy model bazowy

    # Krok 2.2: Budowa pełnego modelu
    model = tf.keras.Sequential([
        # 1. Dodajemy naszą nową warstwę augmentacji NA SAMĄ GÓRĘ
        data_augmentation,
        
        # 2. Reszta modelu jak poprzednio
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # Dropout też pomaga walczyć z przeuczeniem
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Krok 2.3: Kompilacja i PIERWSZY TRENING ("głowy")
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print("Rozpoczynanie WSTĘPNEGO treningu (tylko 'głowa')...")
    history = model.fit(
        train_dataset,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_dataset
    )
    print("Wstępny trening zakończony.")


# --- KROK 3: DOSTRAJANIE (FINE-TUNING) ---
print("Rozpoczynanie DOSTRAJANIA (Fine-Tuning)...")

# "Odmrażamy" model bazowy (teraz jest drugą warstwą, indeks 1)
# model.layers[0] to augmentacja
base_model = model.layers[1] 
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Kontynuujemy trening przez kolejne epoki
history_fine = model.fit(
    train_dataset,
    epochs=FINE_TUNE_EPOCHS, 
    validation_data=validation_dataset
)

print("Dostrajanie zakończone.")

# --- KROK 4: ZAPISYWANIE FINALNEGO MODELU ---
model.save(MODEL_FILENAME)
print(f"Model został pomyślnie zaktualizowany i zapisany w pliku '{MODEL_FILENAME}'")


# --- KROK 5: OCENA MODELU NA DANYCH TESTOWYCH ---
print("\n" + "="*50)
print("ROZPOCZYNANIE FINALNEJ OCENY MODELU NA ZBIORZE TESTOWYM")
print("="*50)

print("Wczytywanie danych testowych...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False 
)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print("Ocenianie modelu...")
results = model.evaluate(test_dataset)

print("\n--- WYNIKI TESTU ---")
print(f"  Strata na danych testowych (Test Loss): {results[0]:.4f}")
print(f"Dokładność na danych testowych (Test Accuracy): {results[1]*100:.2f} %")
print("="*50)
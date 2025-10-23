import os
from ultralytics import YOLO

# --- KONFIGURACJA ---
# Zainstaluj: pip install ultralytics

# Ścieżka do pliku .yaml, który stworzył poprzedni skrypt
base_dir = os.getcwd()
DATA_YAML_PATH = os.path.join(base_dir, "dataset_yolo", "data.yaml")

# Wybór modelu:
# 'yolov8n.pt' - nano (najszybszy, najmniejszy)
# 'yolov8s.pt' - small (dobry balans)
# 'yolov8m.pt' - medium (wolniejszy, dokładniejszy)
MODEL_STARTOWY = 'yolov8m.pt'
# --- KONIEC KONFIGURACJI ---

# 1. Załaduj model YOLO

print(f"Ładowanie bazowego modelu: {MODEL_STARTOWY}")
model = YOLO(MODEL_STARTOWY) 

# 2. Trenuj model na swoich danych
print(f"Rozpoczynam trening na danych z: {DATA_YAML_PATH}")
model.train(
    data=DATA_YAML_PATH, # Ścieżka do pliku konfiguracyjnego
    epochs=200,         
    imgsz=640,          
    batch=8            
)

print("Trening zakończony!")
print("Model, wyniki i wykresy znajdziesz w folderze 'runs/detect/train/'")

# 3. (Opcjonalnie) Uruchom walidację na zbiorze testowym
# YOLO automatycznie robi walidację po każdej epoce,
# ale możemy też uruchomić ją ręcznie na zbiorze 'test'
print("\nUruchamiam finalną walidację na zbiorze TESTOWYM...")
# Znajdź najlepszy zapisany model
sciezka_do_modelu = os.path.join(os.getcwd(), 'runs', 'detect', 'train', 'weights', 'best.pt')

# Załaduj swój najlepszy wytrenowany model
model_finalny = YOLO(sciezka_do_modelu)
metryki = model_finalny.val(split='test')

print("Wyniki na zbiorze testowym:")
print(metryki.box.map)  # To jest najważniejsza metryka (mAP 50-95)

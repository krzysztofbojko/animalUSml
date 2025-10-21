# 🐾 Rozpoznawanie Tropów Zwierząt (YOLOv8)

Projekt ten wykorzystuje model detekcji obiektów **YOLOv8** do identyfikacji i klasyfikacji tropów zwierząt na obrazach. Model został wytrenowany na zbiorze danych **OpenAnimalTracks** i jest w stanie zlokalizować trop na zdjęciu oraz przypisać go do jednego z 18 gatunków.

Celem projektu jest stworzenie "mózgu" dla aplikacji mobilnej, która pomagałaby w rozpoznawaniu tropów napotkanych w terenie.

---

## 🛠️ Użyte Technologie

* Python 3.12
* [Ultralytics YOLOv8](https://ultralytics.com/) (zbudowane na PyTorch)
* [PyYAML](https://pyyaml.org/) (do parsowania plików konfiguracyjnych)
* [tqdm](https://github.com/tqdm/tqdm) (dla pasków postępu)
* [OpenAnimalTracks](https://github.com/Kim-D-K/OpenAnimalTracks) (zbiór danych)

---

## 🚀 Instalacja

1.  **Klonowanie repozytorium:**
    ```bash
    git clone [URL_TWOJEGO_REPOZYTORIUM]
    cd animalUSml
    ```

2.  **(Opcjonalnie) Stworzenie wirtualnego środowiska:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate    # Windows
    ```

3.  **Instalacja zależności:**
    ```bash
    pip install ultralytics pyyaml tqdm
    ```

4.  **Pobranie danych:**
    Rozpakuj plik `OpenAnimalTracks.zip` w głównym folderze projektu (`animalUSml/`).

---

## 🏃 Użycie

Proces uruchomienia projektu składa się z dwóch głównych kroków:

### 1. Przygotowanie Danych (Konwersja COCO -> YOLO)

Posiadany zbiór danych jest w formacie **COCO** (`.json`), a YOLO wymaga formatu `.txt`. Skrypt `konwertuj_dane.py` automatycznie "tłumaczy" adnotacje i przygotowuje całą strukturę folderów.

```bash
# Uruchom ten skrypt tylko raz
python konwertuj_dane.py
```

Ten skrypt stworzy nowy folder `dataset_yolo/`, który zawiera dane treningowe, walidacyjne i testowe w idealnym formacie dla YOLO.

### 2. Trening Modelu

Gdy dane są gotowe, możesz rozpocząć właściwy trening.

```bash
# Uruchom ten skrypt, aby rozpocząć trening
python trenuj_yolo.py
```

Skrypt:
1.  Automatycznie pobierze bazowy model `yolov8s.pt`.
2.  Rozpocznie trening (domyślnie na 100 epok).
3.  Zastosuje mechanizm **EarlyStopping** – jeśli model przestanie się poprawiać, trening zostanie przerwany wcześniej, aby uniknąć przeuczenia.
4.  Po zakończeniu, automatycznie uruchomi walidację na zbiorze testowym.

---

## 🔬 Ewolucja Projektu i Zastosowane Metody

Projekt przeszedł przez kilka etapów i różnych metod, zanim osiągnęliśmy finalne, skuteczne rozwiązanie. Poniżej znajduje się archiwum plików dokumentujących tę ewolucję.

### Etap 1: Klasyfikacja Obrazu (Metoda porzucona)

Początkowo próbowaliśmy nauczyć model **klasyfikacji** (odpowiedzi na pytanie "co jest na tym zdjęciu?"), a nie **detekcji** (odpowiedzi na pytanie "gdzie to jest i co to jest?").

* `apps.py`
    * **Cel:** Główny skrypt treningowy dla modelu klasyfikacji (MobileNetV2 z TensorFlow/Keras).
    * **Problem:** Wymagał idealnie przyciętych zdjęć. Miał ogromny problem z **przeuczeniem (overfitting)** – uczył się na pamięć i osiągał bardzo słabe wyniki (poniżej 30%) na nowych danych.

### Etap 2: Skrypty pomocnicze do Klasyfikacji (Metody porzucone)

Aby walczyć z problemami modelu klasyfikacyjnego, stworzyliśmy skrypty przygotowujące dane.

* `augmentacja.py`
    * **Cel:** Skrypt miał walczyć z przeuczeniem poprzez **fizyczne tworzenie kopii** zdjęć treningowych (obrót o 90, 180, 270 stopni, odbicie lustrzane).
    * **Problem:** Uczyło to model nierealistycznych scenariuszy (np. tropów do góry nogami), co tylko pogorszyło problem przeuczenia.

* `przygotuj_letter_box.py` (w repozytorium jako `przygotuj_letter_box.py`)
    * **Cel:** Lepsze podejście do przygotowania danych. Skrypt dodawał **czarne pasy (letterboxing)** do przyciętych zdjęć, aby stały się kwadratowe bez deformowania proporcji tropu.
    * **Problem:** Mimo że poprawiło to jakość danych wejściowych, podstawowa metoda klasyfikacji w `apps.py` wciąż była niewystarczająco skuteczna.

### Etap 3: Detekcja Obiektów (Aktualna, skuteczna metoda)

Odkryliśmy, że zbiór danych `OpenAnimalTracks` zawierał pliki `.json` z adnotacjami (ramkami). To pozwoliło porzucić całą metodę klasyfikacji i przejść na znacznie potężniejszą **detekcję obiektów** za pomocą YOLO.

* `konwertuj_dane.py`
    * **Status: Aktualny, kluczowy**
    * **Cel:** Jednorazowy skrypt, który **tłumaczy** adnotacje z formatu COCO (`.json`) na format YOLO (`.txt`). Przygotowuje cały folder `dataset_yolo/`.

* `trenuj_yolo.py`
    * **Status: Aktualny, finalny**
    * **Cel:** Główny skrypt treningowy. Używa `ultralytics` do trenowania modelu YOLOv8. To ten skrypt wyprodukował nasz finalny, skuteczny model.

---

## 📂 Struktura Projektu

```
.
├── OpenAnimalTracks/     # Oryginalny, surowy zbiór danych (z .zip)
│   ├── annotations/      # Adnotacje COCO (.json)
│   └── raw_imgs/         # Oryginalne zdjęcia
│
├── dataset_yolo/         # Zbiór danych po konwersji (gotowy dla YOLO)
│   ├── train/            # Podzbiór treningowy (images + labels)
│   ├── val/              # Podzbiór walidacyjny (images + labels)
│   ├── test/             # Podzbiór testowy (images + labels)
│   └── data.yaml         # Plik konfiguracyjny dla YOLO
│
├── runs/                 # Folder tworzony przez YOLO (wyniki, wagi)
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt   # <-- GOTOWY MODEL
│
├── konwertuj_dane.py     # Skrypt do konwersji danych (Krok 1)
├── trenuj_yolo.py        # Główny skrypt treningowy (Krok 2)
│
├── (ARCHIWUM - STARE METODY)
│   ├── apps.py
│   ├── augmentacja.py
│   └── przygotuj_letter_box.py
│
└── README.md             # Ten plik :)
```

---

## 📊 Wyniki

Ostateczny model (wariant `yolov8s`) został wytrenowany przez 177 epok, osiągając najlepsze rezultaty w epoce 77.

Wyniki na **zbiorze testowym** (dane, których model nigdy nie widział):

* **mAP50 (główna dokładność):** `53.2%`
* **mAP50-95 (surowa dokładność):** `34.0%`

### Skuteczność per klasa (mAP50):

| Klasa | Skuteczność |
| :--- | :--- |
| **🏆 Najlepsi:** | |
| goose (gęś) | 80.1% |
| black_bear (niedźwiedź) | 72.9% |
| lion (lew) | 65.4% |
| **😥 Najsłabsi:** | |
| mouse (mysz) | 29.1% |
| beaver (bóbr) | 31.7% |
| elephant (słoń) | 39.5% |

Gotowy do użycia model znajduje się w: `runs/detect/train/weights/best.pt`.

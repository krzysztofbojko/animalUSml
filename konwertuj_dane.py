import json
import os
from tqdm import tqdm # To jest fajny pasek postępu, zainstaluj: pip install tqdm

# --- KONFIGURACJA ---
# Skrypt zakłada, że jest w folderze 'ml-animal'
base_dir = os.getcwd()
annotations_dir = os.path.join(base_dir, "OpenAnimalTracks", "annotations")
raw_imgs_dir = os.path.join(base_dir, "OpenAnimalTracks", "raw_imgs")

# Gdzie zapiszemy nowy, gotowy zestaw danych dla YOLO
output_dir = os.path.join(base_dir, "dataset_yolo")

# Nazwy plików .json
sets_to_process = ["train", "val", "test"]

# --- KONIEC KONFIGURACJI ---

def convert_coco_to_yolo(json_file, img_dir, output_set_dir):
    """
    Tłumaczy adnotacje COCO (.json) na format YOLO (.txt).
    """
    
    # Tworzymy foldery wyjściowe
    output_img_dir = os.path.join(output_set_dir, "images")
    output_label_dir = os.path.join(output_set_dir, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    print(f"Przetwarzam plik: {json_file}")
    
    # Wczytujemy plik .json
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Tworzymy listę nazw klas (WAŻNE)
    category_ids = sorted(categories.keys())
    category_map = {cat_id: i for i, cat_id in enumerate(category_ids)}
    
    # Zapisujemy nazwy klas do pliku .yaml
    if "train" in json_file:
        yaml_data = {
            'train': os.path.join(output_dir, 'train', 'images'),
            'val': os.path.join(output_dir, 'val', 'images'),
            'test': os.path.join(output_dir, 'test', 'images'),
            'nc': len(categories),
            'names': [categories[cat_id] for cat_id in category_ids]
        }
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            import yaml # Potrzebujesz tej biblioteki: pip install pyyaml
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"Zapisano plik konfiguracyjny 'data.yaml' z {len(categories)} klasami.")

    # Przechodzimy przez każdą adnotację (ramkę)
    print("Rozpoczynam konwersję adnotacji...")
    for ann in tqdm(data['annotations']):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox'] # [x_min, y_min, width, height]
        
        # Pobieramy info o obrazku
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Konwersja COCO [x_min, y_min, w, h] na YOLO [x_center_norm, y_center_norm, w_norm, h_norm]
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        w_norm = bbox[2] / img_width
        h_norm = bbox[3] / img_height
        
        # Pobieramy nowy, poprawny ID klasy (0, 1, 2...)
        yolo_cat_id = category_map[cat_id]
        
        # Nazwa pliku .txt musi być taka sama jak nazwa obrazka
        img_filename = img_info['file_name']
        label_filename = os.path.splitext(img_filename.split('/')[-1])[0] + ".txt"
        label_path = os.path.join(output_label_dir, label_filename)
        
        # Zapisujemy linię w pliku .txt
        with open(label_path, 'a') as f:
            f.write(f"{yolo_cat_id} {x_center} {y_center} {w_norm} {h_norm}\n")
            
    # Kopiujemy oryginalne, surowe obrazy do nowego folderu
    print("Kopiowanie oryginalnych obrazów...")
    for img_info in tqdm(images.values()):
        src_path = os.path.join(img_dir, img_info['file_name'])
        dst_path = os.path.join(output_img_dir, img_info['file_name'].split('/')[-1])
        
        # Kopiujemy plik
        import shutil
        shutil.copyfile(src_path, dst_path)

# --- URUCHOMIENIE SKRYPTU ---
for data_set in sets_to_process:
    json_path = os.path.join(annotations_dir, f"{data_set}.json")
    output_set_path = os.path.join(output_dir, data_set)
    
    if os.path.exists(json_path):
        convert_coco_to_yolo(json_path, raw_imgs_dir, output_set_path)
    else:
        print(f"Pominięto: Nie znaleziono pliku {json_path}")

print("\nKonwersja zakończona!")
print(f"Twój zestaw danych gotowy do treningu YOLO znajduje się w: {output_dir}")
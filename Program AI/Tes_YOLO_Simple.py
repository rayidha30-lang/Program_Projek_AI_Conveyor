from ultralytics import YOLO
import cv2

# 1. Tentukan path model dan gambar yang ingin diuji
MODEL_PATH = r"C:\Users\Victus\Downloads\Side_LeftnRight_Conveyor_best (28).pt"
IMAGE_PATH = r"path_ke_gambar_anda.jpg"  # Ganti dengan path file gambar Anda

# 2. Muat model YOLO
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")
print("Classes:", model.names)

# 3. Lakukan inferensi / deteksi objek
results = model.predict(source=IMAGE_PATH, conf=0.25, save=False, show=False)

# 4. Ambil visualisasi hasil dan tampilkan dengan OpenCV
for r in results:
    # r.plot() otomatis menggambar bounding box & label kelas
    annotated_image = r.plot()
    
    # Tampilkan jendela gambar
    cv2.imshow("Hasil Deteksi YOLO", annotated_image)
    cv2.waitKey(0)  # Tekan sembarang tombol di keyboard untuk menutup jendela
    cv2.destroyAllWindows()

    # Cetak hasil koordinat & confidence ke terminal
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        print(f"Detected {cls_name} ({conf*100:.1f}%) at {coords}")

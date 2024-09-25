import cv2
import numpy as np
import os

# Dosya yolunu belirle
image_path = 'C:/Users/omer.turegun/edge_detection/deneme.jpg'

# Çıkış dizinini oluştur
output_dir = 'C:/Users/omer.turegun/edge_detection/output'
os.makedirs(output_dir, exist_ok=True)

# Resmi yükle
image = cv2.imread(image_path)

# Resmin yüklendiğinden emin olun
if image is None:
    print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()


# Resmin boyutlarını al
height, width, _ = image.shape

# Resmi dikey olarak tam ortadan ikiye böl
left_half = image[:, :width // 2]
right_half = image[:, width // 2:]

# Kenar tespiti (Canny kullanarak)
edges_left = cv2.Canny(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY), 100, 200)
edges_right = cv2.Canny(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY), 100, 200)

# İkiye bölünmüş resimleri kaydet
cv2.imwrite(os.path.join(output_dir, 'left_half.jpg'), left_half)
cv2.imwrite(os.path.join(output_dir, 'right_half.jpg'), right_half)

# Kenarları yer değiştirip yatay olarak birleştir
combined_image = np.hstack((right_half, left_half))
cv2.imwrite(os.path.join(output_dir, 'combined_image_swapped.jpg'), combined_image)

# Kenarları yer değiştirip yatay olarak birleştir
combined_edges = np.hstack((edges_right, edges_left))

# Kenar farklarını hesapla (boyutları eşit hale getirin)
if edges_left.shape == edges_right.shape:
    edge_difference = cv2.absdiff(edges_right, edges_left)
else:
    # Boyutları eşitleme
    min_height = min(edges_left.shape[0], edges_right.shape[0])
    min_width = min(edges_left.shape[1], edges_right.shape[1])
    
    edges_left_resized = cv2.resize(edges_left, (min_width, min_height))
    edges_right_resized = cv2.resize(edges_right, (min_width, min_height))
    
    edge_difference = cv2.absdiff(edges_right_resized, edges_left_resized)
    
cv2.imwrite(os.path.join(output_dir, 'edge_difference.jpg'), edge_difference)

# Kenarların temas ettiği noktaları bulma
def find_edge_points(edge_image):
    points = np.column_stack(np.where(edge_image > 0))
    return points

# Sol ve sağ kenarların noktalarını bulma
points_left = find_edge_points(edges_left)
points_right = find_edge_points(edges_right)

# Kenarların temas ettiği noktalar arasındaki mesafeyi hesaplama
def calculate_distance(points1, points2):
    min_distance = float('inf')
    for p1 in points1:
        for p2 in points2:
            distance = np.linalg.norm(p1 - p2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

# Mesafeyi piksel cinsinden hesapla
gap_distance_pixels = calculate_distance(points_left, points_right)

# Piksel başına santimetre değeri (örnek: 0.01 cm/piksel)
pixel_to_cm_ratio = 0.01  # Bu değeri gerçek ölçümlerinize göre ayarlayın
gap_distance_cm = gap_distance_pixels * pixel_to_cm_ratio

print(f"Kenarlar arasındaki en küçük mesafe: {gap_distance_pixels:.2f} piksel, yani {gap_distance_cm:.2f} cm.")

# Kenar farklarının genişliğini orijinal birleşik görüntü ile eşitleyin
edge_difference_resized = cv2.resize(edge_difference, (combined_image.shape[1], combined_image.shape[0]))

# Kenar farklarını içeren birleşik resmi oluştur
final_image_with_edges = np.vstack((combined_image, cv2.cvtColor(edge_difference_resized, cv2.COLOR_GRAY2BGR)))
cv2.imwrite(os.path.join(output_dir, 'final_combined_with_edges.jpg'), final_image_with_edges)

# Sonuçları görselleştir
cv2.imshow("Kenarlar (Sag ve Sol)", combined_edges)
cv2.imshow("Birlesmis Kenar Farklari", edge_difference_resized)

# Tuşa basmayı bekleyin ve ardından pencereyi kapatın
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Resim dikey olarak tam ortadan ikiye bölündü ve {output_dir} dizinine kaydedildi.")
print(f"Yer değiştirilmis ve birleştirilmis resim {output_dir} dizinine kaydedildi.")
print(f"Kenar farklari {output_dir} dizinine kaydedildi.")
print(f"Kenar farklari ile birleşik resim {output_dir} dizinine kaydedildi.")

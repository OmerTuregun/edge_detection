import cv2
import numpy as np

# Fotoğrafı yükle
image = cv2.imread('deneme.jpg', cv2.IMREAD_COLOR)

# Fotoğrafın boyutlarını al
height, width = image.shape[:2]

# Fotoğrafı ikiye böl
left_half = image[:, :width // 2]
right_half = image[:, width // 2:]

# Kenar tespiti uygulama fonksiyonu
def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Özellikleri çıkartma ve eşleştirme fonksiyonu
def find_keypoints_and_descriptors(image):
    edges = apply_edge_detection(image)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(edges, None)
    return keypoints, descriptors

# Eşleştirme fonksiyonu
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Uyum analiz fonksiyonu
def evaluate_matching(keypoints1, keypoints2, matches):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    # Ortalama mesafe hesaplama
    distances = np.sqrt(np.sum((points1 - points2) ** 2, axis=1))
    avg_distance = np.mean(distances)
    return avg_distance

# Kenar tespiti ve özellik çıkartma
left_keypoints, left_descriptors = find_keypoints_and_descriptors(left_half)
right_keypoints, right_descriptors = find_keypoints_and_descriptors(right_half)

# Özellikleri eşleştirme
matches = match_features(left_descriptors, right_descriptors)

# Uyum analizini yapma
average_distance = evaluate_matching(left_keypoints, right_keypoints, matches)

# Eşleşmeleri görselleştirme
def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    combined_image = np.hstack((image1, image2))
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], outImg=combined_image)
    return match_img

left_edges = apply_edge_detection(left_half)
right_edges = apply_edge_detection(right_half)
matched_image = draw_matches(left_edges, left_keypoints, right_edges, right_keypoints, matches)

# Sonuçları göster
cv2.imshow('Matches', matched_image)
cv2.imwrite('matched_image.jpg', matched_image)
print(f"Average Matching Distance: {average_distance:.2f}")
cv2.waitKey(0)
cv2.destroyAllWindows()


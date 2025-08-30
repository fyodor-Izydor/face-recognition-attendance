import cv2
import numpy as np
import face_recognition
import os
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from datetime import datetime

# Gunakan eager execution jika menggunakan TensorFlow 2.x
# tf.compat.v1.enable_eager_execution()

# Inisialisasi daftar gambar dan label
images = []

# Fungsi untuk memuat gambar dan label dari direktori
def load_images_and_labels(path):
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(os.path.join(path, cl))
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames

# Fungsi untuk mendapatkan encoding wajah dari gambar-gambar yang dimuat
def findEncodings(images):
    encodeList = [] # list yang akan berisi semua encoding
    # convert images to RGB
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
        print(encodeList)
    return encodeList

encoded_face_train = findEncodings(images)

print('Proses Encoding Selesai')

# Fungsi untuk mencatat kehadiran dalam file CSV
def mark_attendance(nama, jenisKelamin, usia):
    with open('Attendance.csv', 'a', encoding='utf-8') as f:
        now = datetime.now()
        dtString = now.strftime('%m/%d/%Y,%H:%M:%S')
        f.write(f'{nama},{dtString},{jenisKelamin},{usia} yrs\n')

# Fungsi untuk memprediksi jenis kelamin dan usia dari wajah
def predict_gender_age(current_face_image_blob):
    # Predict gender
    gender_label_list = ['L', 'P']
    gender_protext = "dataset/gender_deploy.prototxt"
    gender_caffemodel = "dataset/gender_net.caffemodel"
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    gender_predictions = gender_cov_net.forward()
    jenisKelamin = gender_label_list[gender_predictions[0].argmax()]

    # Predict age
    age_label_list = ['1 - 10', '11 - 20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '61 - 70', '71 - 80', '81 - 90', '91 - 100']
    age_protext = "dataset/age_deploy.prototxt"
    age_caffemodel = "dataset/age_net.caffemodel"
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    age_predictions = age_cov_net.forward()
    usia = age_label_list[age_predictions[0].argmax()]

    return jenisKelamin, usia

def convert_to_grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def calculate_face_recognition_accuracy(true_labels, predicted_labels):
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total_samples = len(true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy

# Fungsi untuk menilai tingkat kecerahan gambar dan menampilkan pesan
def assess_brightness(gray_img):
    # Hitung rata-rata kecerahan
    mean_brightness = np.mean(gray_img)
    
    # Tentukan rentang kecerahan ahsgdasgdsgaudgsaudushaudhaushduashdusahdshdihasihduwhdiuwahdiauh
    if mean_brightness > 200:
        status = "Terlalu Terang"
    elif mean_brightness < 50:
        status = "Terlalu Gelap"
    else:
        status = "Normal"
    
    return status, mean_brightness

# Fungsi utama program
def main():
    path = 'img/samples'
    images, classNames = load_images_and_labels(path)

    encoded_face_train = findEncodings(images)

    # Load dataset uji yang sesuai (data yang tidak pernah dilihat oleh model sebelumnya)
    path_uji = 'img/uji'
    images_uji, true_labels_uji = load_images_and_labels(path_uji)

    # Mulai membaca video dari webcam
    realtime_cam = cv2.VideoCapture(0)
    mtcnn_detector = MTCNN()
    all_face_locations = []

    if not realtime_cam.isOpened():
        print("Error opening video stream or file")
        

    # Tambahkan variabel boolean untuk menandai apakah data sudah diambil
    data_taken = False

    # Tambahkan list untuk menyimpan hasil prediksi pengenalan wajah
    predicted_labels_uji = []

    # Mendeteksi wajah menggunakan MTCNN dan face_recognition
    while True:
        success, img = realtime_cam.read()
        s_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)

        # Konversi gambar berwarna ke gambar grayscale
        gray_img = convert_to_grayscale(img)

        # Menilai tingkat kecerahan gambar ksajlsdjlasjldsajldsjkd
        status, mean_brightness = assess_brightness(gray_img)

         # Menampilkan matriks nilai grayscale
        print("Nilai Grayscale:")
        print(gray_img)

        # Menambahkan informasi status ke gambar
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, f"Kecerahan: {status}", (10, 30), font, 1, (0, 255, 255), 2)

        # Menampilkan gambar berwarna dan gambar grayscale secara bersamaan
        stacked_img = np.hstack((img, cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Gambar Berwarna vs Gambar Grayscale", stacked_img)

        # # Menampilkan status kecerahan di sudut kiri atas kamera
        # cv2.putText(img, f"Kecerahan: {status}", (10, 50), font, 1, (0, 255, 255), 2)

        # Menampilkan video webcam dengan status kecerahan
        cv2.imshow("Video Webcam", img)

        # Menampilkan gambar berwarna dan gambar grayscale secara bersamaan
        stacked_img = np.hstack((img, cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Gambar Berwarna vs Gambar Grayscale", stacked_img)
        all_face_locations = mtcnn_detector.detect_faces(s_img)
    
        facesCurFrame = face_recognition.face_locations(s_img, model="hog")
        encodesCurFrame = face_recognition.face_encodings(s_img, facesCurFrame)

        for index, current_face_location in enumerate(all_face_locations):
            x, y, width, height = current_face_location['box']
            left_pos = x * 4
            top_pos = y * 4
            right_pos = (x + width) * 4
            bottom_pos = (y + height) * 4
            
            current_face_image = img[top_pos:bottom_pos, left_pos:right_pos]
            AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)

            jenisKelamin, usia = predict_gender_age(current_face_image_blob)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, usia + " " + "yrs", (left_pos, bottom_pos - 10), font, 0.5, (0, 255, 255), 1)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            faceDis = face_recognition.face_distance(encoded_face_train, encodeFace)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.40:
                recognized = 'img/recognized/'
                nama = classNames[matchIndex].upper().lower()
                fileName2 = os.path.join(recognized, nama)
                if not os.path.exists(recognized):
                    os.mkdir(recognized)
                if not os.path.exists(fileName2):
                    os.mkdir(fileName2)
                cv2.imwrite(os.path.join(recognized, nama, f'{nama}.jpg'), img)
                
                if not data_taken:
                    print('Anda diidentifikasi dengan nama:', nama, ', ', jenisKelamin, ', usia:', usia, 'tahun. Data pertama Anda telah disimpan dalam dataframe attendance.csv.')
                    mark_attendance(nama, jenisKelamin, usia)
                    data_taken = True
                
            else:
                nama = 'Unknown'
                print('Wajah Anda', nama, ', tidak dikenali. Data Anda tidak dapat disimpan dalam dataframe attendance.csv.')

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (128, 0, 128), cv2.FILLED)
            cv2.putText(img, nama, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Video Webcam", img)

        # k = cv2.waitKey(1)
        # if k == 27:
        #     break

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    face_recognition_accuracy = calculate_face_recognition_accuracy(true_labels_uji, predicted_labels_uji)

    print("Akurasi Pengenalan Wajah:", face_recognition_accuracy)
    

    realtime_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

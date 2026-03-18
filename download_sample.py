import os
import urllib.request
import time

def download_video(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def main():
    raw_dir = os.path.join('dataset', 'raw')
    
    # We will create four 'people' directories and put the same/similar videos in them just to test the pipeline runs
    person_a_dir = os.path.join(raw_dir, 'person_A')
    person_b_dir = os.path.join(raw_dir, 'person_B')
    person_c_dir = os.path.join(raw_dir, 'person_C')
    person_d_dir = os.path.join(raw_dir, 'person_D')
    
    os.makedirs(person_a_dir, exist_ok=True)
    os.makedirs(person_b_dir, exist_ok=True)
    os.makedirs(person_c_dir, exist_ok=True)
    os.makedirs(person_d_dir, exist_ok=True)
    
    # Using generic sample footages
    # Person A
    url_a1 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
    url_a2 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    
    # Person B
    url_b1 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4"
    url_b2 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"

    # Person C
    url_c1 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4"
    url_c2 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4"
    
    # Person D
    url_d1 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4"
    url_d2 = "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4"
    
    download_video(url_a1, os.path.join(person_a_dir, "video1.mp4"))
    download_video(url_a2, os.path.join(person_a_dir, "video2.mp4"))
    
    download_video(url_b1, os.path.join(person_b_dir, "video1.mp4"))
    download_video(url_b2, os.path.join(person_b_dir, "video2.mp4"))

    download_video(url_c1, os.path.join(person_c_dir, "video1.mp4"))
    download_video(url_c2, os.path.join(person_c_dir, "video2.mp4"))
    
    download_video(url_d1, os.path.join(person_d_dir, "video1.mp4"))
    download_video(url_d2, os.path.join(person_d_dir, "video2.mp4"))

if __name__ == "__main__":
    main()

import os, sys, cv2
import tqdm
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
from utils.data_collection import get_files

def main(img_folder):
    files = get_files(img_folder, "jpg")
    if(not files):
        return
    h , w , c = cv2.imread(files[0]).shape
    video_name = os.path.join(os.path.dirname(img_folder), os.path.basename(img_folder) + ".mp4")
    video = cv2.VideoWriter(video_name, 0x7634706d , 30, (w,h)) # 30 fps
    for f in tqdm.tqdm(files):
        img = cv2.imread(f)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    img_folder = "C:/Users/Jessica/Documents/Proyecto_ssd/datasets/vrenv_test/close_open_out"
    main(img_folder)
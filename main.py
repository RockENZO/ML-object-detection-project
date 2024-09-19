from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/yolov5su.pt')

    save_video(video_frames,'output_videos/output_video.avi')

if __name__ == "__main__":
    main()
import yt_dlp
import sys

def download_youtube_mp4(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'outtmpl': './%(title)s.%(ext)s',  # 저장 경로 및 파일명 템플릿
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # mp4 비디오+오디오 우선 선택
        'merge_output_format': 'mp4',  # 병합 후 포맷 mp4 지정
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python youtube_downloader.py <video_id>")
        sys.exit(1)
    video_id = sys.argv[1]
    download_youtube_mp4(video_id)
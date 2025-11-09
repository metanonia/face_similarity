use opencv::{
    core,
    highgui,
    prelude::*,
    videoio,
    Result,
};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_file_path>", args[0]);
        std::process::exit(1);
    }

    let video_path = &args[1];
    let mut cap = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("비디오 파일을 열 수 없습니다: {}", video_path);
    }

    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.size()?.width == 0 {
            break; // 비디오 끝
        }
        highgui::imshow("Video Playback", &frame)?;
        let key = highgui::wait_key(30)?;
        if key == 27 {
            break; // ESC 키로 종료
        }
    }

    Ok(())
}

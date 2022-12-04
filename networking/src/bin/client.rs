use std::io::prelude::*;
use std::net::TcpStream;
use std::time::{Duration, SystemTime};

const IMAGE_DIR: &str = "images";
const LABEL_DIR: &str = "labels";
const WEIGHT_DIR: &str = "weights";

//const SERVER_ADDR: &str = "100.122.154.52:7887";
const SERVER_ADDR: &str = "10.28.70.33:7887";

fn main() {
    let mut init_time: u64 = 0;
    let mut args_iter = std::env::args();
    if args_iter.len() >= 2 {
        args_iter.next();
        let arg = args_iter.next().unwrap();
        if arg == "now" {
            init_time = SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
        }
        else {
            init_time = arg.parse().unwrap();
        }
    }
    std::fs::create_dir_all(IMAGE_DIR).unwrap();
    std::fs::create_dir_all(LABEL_DIR).unwrap();
    std::fs::create_dir_all(WEIGHT_DIR).unwrap();

    loop {
        let res = TcpStream::connect(SERVER_ADDR);
        if res.is_err() {
            println!("Could not connect to server, retrying...");
            std::thread::sleep(Duration::from_secs(1));
        }
        else {
            let stream = res.unwrap();
            let write_stream = stream.try_clone().expect("Socket clone failed");
            let read_stream = stream.try_clone().expect("Socket clone failed");

            let write_thread_handle = std::thread::spawn(move || write_images(init_time, write_stream));
            let read_thread_handle = std::thread::spawn(move || read_weights(read_stream));
            write_thread_handle.join().unwrap();
            read_thread_handle.join().unwrap();
        }
    }

} // the stream is closed here

fn write_images(init_time: u64, mut stream: TcpStream) {
    let mut last_time = init_time;
    println!("In write_images with init time {last_time}");
    loop {
        let data = networking::look_for_images(IMAGE_DIR, LABEL_DIR, Some(last_time)).unwrap();
        let data_len = data.len() as u32;
        stream.write_all(&data_len.to_be_bytes()).unwrap();
        for (image, label, time) in data {
            stream.write_all(&time.to_be_bytes()).unwrap();
            let width = image.width();
            let height = image.height();
            let image_raw = image.into_raw();
            let label_raw = label.into_bytes();
            stream.write_all(&width.to_be_bytes()).unwrap();
            stream.write_all(&height.to_be_bytes()).unwrap();
            // Cast to uint32 as usize is architecture-dependent
            stream.write_all(&(image_raw.len() as u32).to_be_bytes()).unwrap();
            stream.write_all(&image_raw).unwrap();
            stream.write_all(&(label_raw.len() as u32).to_be_bytes()).unwrap();
            stream.write_all(&label_raw).unwrap();
            // Update last_time
            last_time = std::cmp::max(last_time, time);
        }
        stream.flush().unwrap();
        std::thread::sleep(Duration::from_secs(1));
    }    
}

fn read_weights(mut stream: TcpStream) {
    loop {
        let mut heartbeat = [0u8];
        stream.read_exact(&mut heartbeat).unwrap();
        if heartbeat[0] == 0 {
            continue;
        }

        let mut size_buf = [0u8; 4];
        stream.read_exact(&mut size_buf).unwrap();
        let weight_size = u32::from_be_bytes(size_buf);
        let mut weight_buf = Vec::new();
        weight_buf.resize(weight_size as usize, 0);
        stream.read_exact(&mut weight_buf).unwrap();

        let weight_path = format!("{}/model.tflite", WEIGHT_DIR);
        std::fs::write(weight_path, weight_buf).unwrap();
        println!("Read new weights");
    }
}

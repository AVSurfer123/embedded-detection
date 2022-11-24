use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, SystemTime};

const IMAGE_DIR: &str = "server_images";
const LABEL_DIR: &str = "server_labels";
const WEIGHT_DIR: &str = "server_weights";

fn main(){
    let listener = TcpListener::bind("127.0.0.1:7887").expect("Couldn't bind to address");
    // accept connections and process them in separate threads
    for stream in listener.incoming() {
        std::thread::spawn(|| handle_client(&mut stream.expect("Invalid TCP connection")));
    }
}

fn handle_client(stream: &mut TcpStream) {
    println!("Connection from {:?}", stream.peer_addr().unwrap());
    
    let mut read_stream = stream.try_clone().expect("Socket clone failed");
    let mut write_stream = stream.try_clone().expect("Socket clone failed");

    std::thread::spawn(move || read_images(&mut read_stream));
    std::thread::spawn(move || write_weights(&mut write_stream));
}

fn read_images(stream: &mut TcpStream) {
    loop {
        let mut data_len = [0u8; 4];
        stream.read_exact(&mut data_len).unwrap();
        let data_len = u32::from_be_bytes(data_len);
        for _ in 0..data_len {
            // Read time
            let mut time_buf = [0u8; 8];
            stream.read_exact(&mut time_buf).unwrap();
            let time = u64::from_be_bytes(time_buf);

            let mut size_buf = [0u8; 4];
            // Read width, height
            stream.read_exact(&mut size_buf).unwrap();
            let width = u32::from_be_bytes(size_buf);
            stream.read_exact(&mut size_buf).unwrap();
            let height = u32::from_be_bytes(size_buf);
            
            // Read image
            stream.read_exact(&mut size_buf).unwrap();
            let image_len = u32::from_be_bytes(size_buf);
            let mut image_buf = Vec::new();
            image_buf.resize(image_len as usize, 0);
            stream.read_exact(&mut image_buf).unwrap();

            // Read label
            stream.read_exact(&mut size_buf).unwrap();
            let label_len = u32::from_be_bytes(size_buf);
            let mut label_buf = Vec::new();
            label_buf.resize(label_len as usize, 0);
            stream.read_exact(&mut label_buf).unwrap();

            let image_path = format!("{}/{}.png", IMAGE_DIR, time.to_string());
            image::save_buffer(image_path, &image_buf, width, height, image::ColorType::Rgb8).unwrap();  
            let label_path = format!("{}/{}.txt", LABEL_DIR, time.to_string());
            std::fs::write(label_path, label_buf).unwrap();
            println!("Read image");
        }
    }
}

fn write_weights(stream: &mut TcpStream) {
    let mut last_time = std::time::UNIX_EPOCH;
    loop {
        match networking::get_recent_weight(WEIGHT_DIR, last_time) {
            Some(weights) => {
                stream.write_all(&[1u8]).unwrap();
                stream.write_all(&(weights.len() as u32).to_be_bytes()).unwrap();
                stream.write_all(&weights).unwrap();
                stream.flush().unwrap();
                last_time = SystemTime::now();
                println!("Wrote weights");
            }
            None => {
                stream.write_all(&[0u8]).unwrap();
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}


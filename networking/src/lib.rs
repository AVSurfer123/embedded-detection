use std::fs;
use std::io;
use std::time::SystemTime;

// Time to wait after image/model is created to ensure it is fully written before reading
const IMAGE_READ_DELAY: f32 = 1.0;
const WEIGHT_READ_DELAY: f32 = 1.0;

pub fn look_for_images(image_dir_name: &str, label_dir_name: &str, last_time: Option<u64>) -> io::Result<Vec<(image::RgbImage, String, u64)>> {
    let last_time = last_time.unwrap_or(0);
    let cur_time = SystemTime::now();
    let mut data = Vec::new();
    for result in fs::read_dir(image_dir_name)? {
        let entry = &result?;
        let path_name = entry.path();
        // let img_time = entry.metadata()?.modified()?.duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
        let file_name = path_name.file_stem().unwrap().to_str().unwrap();
        let img_time: u64 = file_name.parse().expect("File name isn't a timestamp");
        if img_time > last_time && (cur_time.duration_since(entry.metadata()?.modified()?).unwrap().as_secs_f32() > IMAGE_READ_DELAY) {
            println!("Adding image {} at time {img_time}", path_name.to_str().unwrap());
            let img = image::open(&path_name).unwrap().into_rgb8();
            let label = fs::read_to_string(format!("{}/{}.txt", label_dir_name, file_name))?;
            data.push((img, label, img_time));
        }
    }
    Ok(data)
}

pub fn get_recent_weight(weight_dir_name: &str, last_time: SystemTime) -> Option<Vec<u8>> {
    fs::read_dir(weight_dir_name).unwrap()
        .max_by_key(|e| 
            e.as_ref().unwrap().metadata().unwrap().modified().unwrap()
        ).filter(|e| {
            let file_time = e.as_ref().unwrap().metadata().unwrap().modified().unwrap();
            file_time > last_time && SystemTime::now().duration_since(file_time).unwrap().as_secs_f32() > WEIGHT_READ_DELAY
        }).map(|e| {
            let path = e.unwrap().path();
            println!("Weight path: {path:?}");
            fs::read(path).unwrap()
        })
}

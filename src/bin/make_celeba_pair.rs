use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use rand::prelude::IndexedRandom;

fn main() -> std::io::Result<()> {
    let image_dir = "./CelebA/Img/img_celeba";
    let identity_file = "./CelebA/Anno/Identity_CelebA.txt";

    // Identity 파일 파싱: 이미지 파일명 -> ID 매핑
    let file = File::open(identity_file)?;
    let reader = BufReader::new(file);

    let mut id_map: HashMap<u32, Vec<String>> = HashMap::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let filename = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<u32>() {
                    id_map.entry(id).or_default().push(filename);
                }
            }
        }
    }

    // 동일인 쌍 생성 (모든 조합)
    let mut positive_pairs = Vec::new();
    for (_id, images) in id_map.iter() {
        if images.len() > 1 {
            for i in 0..images.len() {
                for j in (i + 1)..images.len() {
                    positive_pairs.push((images[i].clone(), images[j].clone()));
                }
            }
        }
    }

    println!("Total positive pairs (same person): {}", positive_pairs.len());

    // 타인 쌍 생성 (동일인 쌍 갯수만큼)
    let all_ids: Vec<&u32> = id_map.keys().collect();
    let mut negative_pairs = Vec::new();
    let mut rng = rand::thread_rng();

    while negative_pairs.len() < positive_pairs.len() {
        let ids = all_ids.choose_multiple(&mut rng, 2).cloned().collect::<Vec<&u32>>();
        if ids.len() < 2 {
            break;
        }
        if let (Some(img1), Some(img2)) = (
            id_map.get(ids[0]).unwrap().choose(&mut rng),
            id_map.get(ids[1]).unwrap().choose(&mut rng),
        ) {
            negative_pairs.push((img1.clone(), img2.clone()));
        }
    }

    println!("Total negative pairs (different persons): {}", negative_pairs.len());

    // 파일로 저장
    let mut pos_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("positive_pairs.txt")?;

    for (img1, img2) in &positive_pairs {
        writeln!(pos_file, "{},{}", img1, img2)?;
    }

    let mut neg_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("negative_pairs.txt")?;

    for (img1, img2) in &negative_pairs {
        writeln!(neg_file, "{},{}", img1, img2)?;
    }

    Ok(())
}

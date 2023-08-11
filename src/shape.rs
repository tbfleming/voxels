use bevy::prelude::UVec3;

use crate::*;

pub fn sphere(size: u32, material: u8) -> VoxelGridVec {
    let mut data = Vec::new();
    let size = size as i32;
    data.resize(((size + 2) * (size + 2) * (size + 2)) as usize, 0);
    let radius = (size as f32) / 2.0;
    let dist_squared = |x, y, z| {
        let dx = x - radius;
        let dy = y - radius;
        let dz = z - radius;
        dx * dx + dy * dy + dz * dz
    };
    // let dist_squared_2d = |x, y| {
    //     let dx = x - radius;
    //     let dy = y - radius;
    //     dx * dx + dy * dy
    // };
    let inside = |x, y, z| dist_squared(x + 0.5, y + 0.5, z + 0.5) < radius * radius;
    let index =
        |x, y, z| ((x + 1) + (y + 1) * (size + 2) + (z + 1) * (size + 2) * (size + 2)) as usize;

    for z in 0..size + 1 {
        // let radius_2d =
        //     (radius * radius - (z as f32 - radius) * (z as f32 - radius)).sqrt();
        for y in 0..size + 1 {
            for x in 0..size + 1 {
                if inside(x as f32, y as f32, z as f32) {
                    data[index(x, y, z)] |= (material as u32) << 24;
                }
                let other_inside =
                    |dx, dy, dz| inside((x + dx) as f32, (y + dy) as f32, (z + dz) as f32);
                let count = other_inside(-1, -1, -1) as u32
                    + other_inside(-1, -1, 0) as u32
                    + other_inside(-1, 0, -1) as u32
                    + other_inside(-1, 0, 0) as u32
                    + other_inside(0, -1, -1) as u32
                    + other_inside(0, -1, 0) as u32
                    + other_inside(0, 0, -1) as u32
                    + other_inside(0, 0, 0) as u32;
                if count != 0 && count != 8 {
                    let factor = radius / dist_squared(x as f32, y as f32, z as f32).sqrt();
                    let delta = |p| {
                        (((p as f32 - radius) * factor + radius - p as f32) * 64.0)
                            .round()
                            .clamp(-127.0, 127.0) as i8 as u8 as u32
                    };
                    data[index(x, y, z)] |= delta(x) | (delta(y) << 8) | (delta(z) << 16);
                }
            }
        }
    }
    let size = size as u32;
    VoxelGridVec {
        size: UVec3::new(size, size, size),
        data,
    }
}

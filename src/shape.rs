use bevy::prelude::UVec3;

use crate::*;

pub fn sphere(size: u32, material: u8) -> VoxelGridVec {
    let mut data = Vec::new();
    let size = size as i32;
    data.resize(((size + 2) * (size + 2) * (size + 2)) as usize, 0);
    let half_size = (size as f32) / 2.0;
    let dist_squared = |x, y, z| {
        let dx = x as f32 - half_size;
        let dy = y as f32 - half_size;
        let dz = z as f32 - half_size;
        dx * dx + dy * dy + dz * dz
    };
    let dist_squared_2d = |x, y| {
        let dx = x as f32 - half_size;
        let dy = y as f32 - half_size;
        dx * dx + dy * dy
    };
    let inside = |x, y, z| dist_squared(x, y, z) < half_size * half_size;
    let index =
        |x, y, z| ((x + 1) + (y + 1) * (size + 2) + (z + 1) * (size + 2) * (size + 2)) as usize;

    for z in -1..size + 1 {
        let radius_2d =
            (half_size * half_size - (z as f32 - half_size) * (z as f32 - half_size)).sqrt();
        for y in -1..size + 1 {
            for x in -1..size + 1 {
                if inside(x, y, z) {
                    data[index(x, y, z)] |= (material as u32) << 24;
                }
                let other_inside = |dx, dy, dz| inside(x + dx, y + dy, z + dz);
                let count = other_inside(-1, -1, -1) as u32
                    + other_inside(-1, -1, 0) as u32
                    + other_inside(-1, 0, -1) as u32
                    + other_inside(-1, 0, 0) as u32
                    + other_inside(0, -1, -1) as u32
                    + other_inside(0, -1, 0) as u32
                    + other_inside(0, 0, -1) as u32
                    + other_inside(0, 0, 0) as u32;
                if count != 0 && count != 8 {
                    let factor = radius_2d / dist_squared_2d(x, y).sqrt();
                    let delta = |p| {
                        (((p as f32 - half_size) * factor + half_size - p as f32) * 64.0)
                            .round()
                            .clamp(-127.0, 127.0) as i8 as u8 as u32
                    };
                    data[index(x, y, z)] |= delta(x) | (delta(y) << 8);
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

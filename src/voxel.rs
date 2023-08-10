use bevy::{
    core::cast_slice,
    ecs::query::Has,
    prelude::*,
    reflect::TypePath,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::*,
        renderer::RenderDevice,
        Render, RenderApp, RenderSet,
    },
};
use std::sync::{Arc, Mutex};

pub(crate) const WGSL_VOXEL_GRID_IN_SIZE_BINDING: u32 = 2;
pub(crate) const WGSL_VOXEL_GRID_IN_BINDING: u32 = 3;

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<VoxelGridData>::default());
        app.add_plugins(ExtractComponentPlugin::<VoxelGridStorageBuffer>::default());
        app.add_plugins(ExtractComponentPlugin::<CopyVoxelGridToStorageBuffer>::default());
        app.add_systems(First, finalize_copy_data_to_storage);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, copy_data_to_storage.in_set(RenderSet::Prepare));
    }
}

// Each voxel is 4 bytes:
// * Byte 0: i8: offset_x * 64
// * Byte 1: i8: offset_y * 64
// * Byte 2: i8: offset_z * 64
// * Byte 3: u8: material. 0 means empty.
//
// The offsets modify the voxel's lower-left corner and have range (-2.0, 2.0),
// where 1.0 is the distance between voxel centers. If an offset value is 0x80,
// it is treated as 0x81.
//
// The voxels are packed by x, then y, then z. Each dimension is padded on both
// sides by 1 voxel. The offsets at the start padding don't matter. The
// offsets at the end padding complete the voxel bounds. Non-0 material in padding
// excludes the faces at the edges of the voxel grid.
//
// lock order: VoxelGridData, VoxelGridStorageBuffer
#[derive(Component, Clone, Debug, TypePath, ExtractComponent)]
pub struct VoxelGridData {
    pub size: UVec3,
    pub data: Arc<Mutex<Option<Vec<u32>>>>,
}

impl VoxelGridData {
    pub fn without_data(size: UVec3) -> Self {
        Self {
            size,
            data: Arc::new(Mutex::new(None)),
        }
    }

    pub fn new(size: UVec3, material: u8) -> Self {
        let mut data = Vec::new();
        data.resize(((size.x + 2) * (size.y + 2) * (size.z + 2)) as usize, 0);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    data[((x + 1) + (y + 1) * (size.x + 2) + (z + 1) * (size.x + 2) * (size.y + 2))
                        as usize] = (material as u32) << 24;
                }
            }
        }
        Self {
            size,
            data: Arc::new(Mutex::new(Some(data))),
        }
    }

    pub fn sphere(size: u32, material: u8) -> Self {
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
        Self {
            size: UVec3::new(size, size, size),
            data: Arc::new(Mutex::new(Some(data))),
        }
    }
}

// lock order: VoxelGridData, VoxelGridStorageBuffer
#[derive(Component, Clone, Debug, TypePath, ExtractComponent)]
pub struct VoxelGridStorageBuffer {
    pub size: UVec3,
    pub buffer: Arc<Mutex<Option<Buffer>>>,
}

impl VoxelGridStorageBuffer {
    pub fn new(size: UVec3) -> Self {
        Self {
            size,
            buffer: default(),
        }
    }
}

#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct CopyVoxelGridToStorageBuffer;

fn copy_data_to_storage(
    render_device: Res<RenderDevice>,
    query: Query<(&VoxelGridData, &VoxelGridStorageBuffer), Has<CopyVoxelGridToStorageBuffer>>,
) {
    for (voxel_grid_data, voxel_grid_storage_buffer) in query.iter() {
        if voxel_grid_data.size != voxel_grid_storage_buffer.size {
            println!("** copy_data_to_storage: size mismatch");
            continue;
        }
        let data = voxel_grid_data.data.lock().unwrap();
        let mut buffer = voxel_grid_storage_buffer.buffer.lock().unwrap();
        if buffer.is_some() {
            continue;
        }
        let Some(data) = &*data else {
            println!("** copy_data_to_storage: no data");
            continue;
        };
        println!("** copy_data_to_storage");
        let storage_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: (data.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        storage_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice::<u32, u8>(data));
        // println!("{:?}\n=====", data);
        // println!(
        //     "{:?}",
        //     cast_slice::<u8, u32>(&storage_buffer.slice(..).get_mapped_range())
        // );
        storage_buffer.unmap();
        *buffer = Some(storage_buffer);
    }
}

fn finalize_copy_data_to_storage(
    mut commands: Commands,
    query: Query<(Entity, &VoxelGridStorageBuffer), Has<CopyVoxelGridToStorageBuffer>>,
) {
    for (entity, voxel_grid_storage_buffer) in query.iter() {
        let buffer = voxel_grid_storage_buffer.buffer.lock().unwrap();
        if buffer.is_some() {
            commands
                .entity(entity)
                .remove::<CopyVoxelGridToStorageBuffer>();
        }
    }
}

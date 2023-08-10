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
        app.add_plugins(ExtractComponentPlugin::<VoxelGrid>::default());
        app.add_plugins(ExtractComponentPlugin::<CopyDataToVoxelGrid>::default());
        app.add_systems(First, finalize_copy_data_to_voxel_grid);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, copy_data_to_voxel_grid.in_set(RenderSet::Prepare));
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
// lock order: VoxelGridData, VoxelGrid
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
}

// lock order: VoxelGridData, VoxelGrid
#[derive(Component, Clone, Debug, TypePath, ExtractComponent)]
pub struct VoxelGrid {
    pub size: UVec3,
    pub buffer: Arc<Mutex<Option<Buffer>>>,
}

impl VoxelGrid {
    pub fn new(size: UVec3) -> Self {
        Self {
            size,
            buffer: default(),
        }
    }
}

#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct CopyDataToVoxelGrid;

fn copy_data_to_voxel_grid(
    render_device: Res<RenderDevice>,
    query: Query<(&VoxelGridData, &VoxelGrid), Has<CopyDataToVoxelGrid>>,
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

fn finalize_copy_data_to_voxel_grid(
    mut commands: Commands,
    query: Query<(Entity, &VoxelGrid), Has<CopyDataToVoxelGrid>>,
) {
    for (entity, voxel_grid_storage_buffer) in query.iter() {
        let buffer = voxel_grid_storage_buffer.buffer.lock().unwrap();
        if buffer.is_some() {
            commands.entity(entity).remove::<CopyDataToVoxelGrid>();
        }
    }
}

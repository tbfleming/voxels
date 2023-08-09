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

#[derive(Component, Clone, Debug, TypePath, ExtractComponent)]
pub struct VoxelGridData {
    pub size: Vec3,
    pub data: Arc<Mutex<Option<Vec<u32>>>>,
}

#[derive(Component, Clone, Debug, TypePath, ExtractComponent)]
pub struct VoxelGridStorageBuffer {
    pub size: Vec3,
    pub buffer: Arc<Mutex<Option<Buffer>>>,
}

#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct CopyVoxelGridToStorageBuffer;

fn copy_data_to_storage(
    render_device: Res<RenderDevice>,
    query: Query<(&VoxelGridData, &VoxelGridStorageBuffer), Has<CopyVoxelGridToStorageBuffer>>,
) {
    // println!("** copy_data_to_storage");
    for (voxel_grid_data, voxel_grid_storage_buffer) in query.iter() {
        // println!("** copy_data_to_storage: ?");
        if voxel_grid_data.size != voxel_grid_storage_buffer.size {
            continue;
        }
        let data = voxel_grid_data.data.lock().unwrap();
        let mut buffer = voxel_grid_storage_buffer.buffer.lock().unwrap();
        if buffer.is_some() {
            continue;
        }
        let Some(data) = &*data else {continue;};
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

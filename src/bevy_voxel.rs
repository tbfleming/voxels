use bevy::{
    ecs::query::Has,
    prelude::*,
    reflect::TypePath,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        renderer::RenderDevice,
        Render, RenderApp, RenderSet,
    },
};
use std::sync::{Arc, Mutex};

use crate::*;

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

// lock order: SharedVoxelGridContent, SharedVoxelGridBuffer
pub type SharedVoxelGridContent = Arc<Mutex<Option<VoxelGridVec>>>;

#[derive(Component, Clone, Default, Debug, TypePath, ExtractComponent, Deref, DerefMut)]
pub struct VoxelGridData(SharedVoxelGridContent);

impl VoxelGridData {
    pub fn new() -> Self {
        default()
    }
}

impl From<VoxelGridVec> for VoxelGridData {
    fn from(value: VoxelGridVec) -> Self {
        VoxelGridData(Arc::new(Mutex::new(Some(value))))
    }
}

// lock order: SharedVoxelGridContent, SharedVoxelGridBuffer
pub type SharedVoxelGridBuffer = Arc<Mutex<Option<VoxelGridBuffer>>>;

#[derive(Component, Clone, Debug, Default, TypePath, ExtractComponent, Deref, DerefMut)]
pub struct VoxelGrid(SharedVoxelGridBuffer);

impl VoxelGrid {
    pub fn new() -> Self {
        default()
    }
}

impl From<VoxelGridBuffer> for VoxelGrid {
    fn from(value: VoxelGridBuffer) -> Self {
        VoxelGrid(Arc::new(Mutex::new(Some(value))))
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
        let data_lock = voxel_grid_data.lock().unwrap();
        let mut vg_buffer_lock = voxel_grid_storage_buffer.lock().unwrap();
        if vg_buffer_lock.is_some() {
            continue;
        }
        let Some(data) = &*data_lock else {
            println!("** copy_data_to_storage: no data");
            continue;
        };
        println!("** copy_data_to_storage");
        *vg_buffer_lock = Some(VoxelGridBuffer::from_content(
            data,
            render_device.wgpu_device(),
        ));
    }
}

fn finalize_copy_data_to_voxel_grid(
    mut commands: Commands,
    query: Query<(Entity, &VoxelGrid), Has<CopyDataToVoxelGrid>>,
) {
    for (entity, voxel_grid_storage_buffer) in query.iter() {
        let vg_buffer = voxel_grid_storage_buffer.lock().unwrap();
        if vg_buffer.is_some() {
            commands.entity(entity).remove::<CopyDataToVoxelGrid>();
        }
    }
}

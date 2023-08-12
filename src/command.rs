use glam::{UVec3, Vec3};
use std::sync::{Arc, Mutex};
use wgpu::{BindGroupLayout, BufferAsyncError, CommandEncoder, ComputePipeline, Device};

use crate::voxel::*;

// lock order: SharedVoxelGridContent, SharedVoxelGridBuffer
pub type SharedVoxelGridBuffer = Arc<Mutex<Option<VoxelGridBuffer>>>;

/// A command to be executed
///
/// Call the following in order:
/// * `[generate_mesh_bind_group_layout]` once per device.
///    After this, you should generate a pipeline using the
///    layout then cache both the layout and the pipeline.
///    You'll need these to call `[prepare]` and `[add_pass]`.
/// * `[prepare]`
/// * `[add_pass]`
/// * `[add_copy]`. This may be on a different queue, but the
///   copy's execution must happen after the pass's execution.
/// * `[async_finish]`. Only call this after the pass and copy
///   operations have finished executing on the GPU.
trait Command {
    /// Shader entry point
    const ENTRY_POINT: &'static str;

    /// Create bind group layout
    fn generate_mesh_bind_group_layout(device: &Device) -> BindGroupLayout;

    /// Create buffers and bind group. get_bind_group_layout's argument
    /// is `[ENTRY_POINT]`.
    fn prepare<'a, F: FnOnce(&str) -> &'a BindGroupLayout>(
        &mut self,
        device: &Device,
        get_bind_group_layout: F,
    );

    /// Add the compute pass to the command encoder. get_pipeline's argument
    /// is `[ENTRY_POINT]`.
    fn add_pass<'a, F: FnOnce(&str) -> &'a ComputePipeline>(
        &self,
        encoder: &mut CommandEncoder,
        get_pipeline: F,
    );

    /// Add buffer copies, if any, to the command encoder
    fn add_copy(&self, encoder: &mut CommandEncoder);

    /// Map the copy buffers if needed and perform any finalization steps, then call the callback
    fn async_finish(&mut self, done: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static);
}

/// Convert a voxel grid to a mesh.
#[derive(Debug)]
pub struct GenerateMeshCommand<F> {
    /// Grid to turn into a mesh
    pub grid: SharedVoxelGridBuffer,

    /// Receives the generated vertexes and normals
    pub receive_result: F,

    cmd_impl: Option<GenerateMeshImpl>,
}

impl<F: FnMut(Vec<Vec3>, Vec<Vec3>) + 'static + Send + Clone> GenerateMeshCommand<F> {
    pub fn new(grid: SharedVoxelGridBuffer, receive_result: F) -> Self {
        Self {
            grid,
            receive_result,
            cmd_impl: Default::default(),
        }
    }
}

impl<F: FnMut(Vec<Vec3>, Vec<Vec3>) + 'static + Send + Clone> Command for GenerateMeshCommand<F> {
    const ENTRY_POINT: &'static str = GENERATE_MESH_ENTRY_POINT;

    fn generate_mesh_bind_group_layout(device: &Device) -> BindGroupLayout {
        generate_mesh_bind_group_layout(device)
    }

    fn prepare<'a, G: FnOnce(&str) -> &'a BindGroupLayout>(
        &mut self,
        device: &Device,
        get_bind_group_layout: G,
    ) {
        let lock = self.grid.lock().unwrap();
        self.cmd_impl = Some(GenerateMeshImpl::new(
            device,
            get_bind_group_layout(Self::ENTRY_POINT),
            lock.as_ref().expect("Missing grid in GenerateMeshCommand"),
        ));
    }

    fn add_pass<'a, G: FnOnce(&str) -> &'a ComputePipeline>(
        &self,
        encoder: &mut CommandEncoder,
        get_pipeline: G,
    ) {
        self.cmd_impl
            .as_ref()
            .unwrap()
            .add_pass(get_pipeline(Self::ENTRY_POINT), encoder);
    }

    /// Add buffer copies, if any, to the command encoder
    fn add_copy(&self, encoder: &mut CommandEncoder) {
        self.cmd_impl.as_ref().unwrap().add_copy(encoder);
    }

    /// Map the copy buffers if needed (async) then call the callback
    fn async_finish(&mut self, done: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static) {
        let mut receive_result = self.receive_result.clone();
        self.cmd_impl
            .take()
            .unwrap()
            .async_map_buffer(move |cmd_impl, res| {
                if res.is_ok() {
                    let (m, n) = cmd_impl.get_mesh();
                    receive_result(m, n);
                    done(Ok(()));
                } else {
                    done(res);
                }
            });
    }
}

/// Create a voxel grid with the given size.
#[derive(Clone, Debug, Default)]
pub struct CreateGridCommand {
    /// Destination. Reuse the existing buffer if it already exists and its size matches.
    grid: SharedVoxelGridBuffer,

    /// Size of the voxel grid, excluding padding
    size: UVec3,

    /// If Some, clear the voxel grid to this material
    clear_material: Option<u8>,
}

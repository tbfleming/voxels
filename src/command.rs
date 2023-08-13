use glam::{IVec3, UVec3, Vec3};
use parking_lot::Mutex;
use std::{fmt::Debug, sync::Arc};
use wgpu::{BindGroupLayout, BufferAsyncError, CommandEncoder, ComputePipeline, Device};

use crate::voxel::*;

// lock order: SharedVoxelGridContent, SharedVoxelGridBuffer
pub type SharedVoxelGridBuffer = Arc<Mutex<Option<VoxelGridBuffer>>>;

/// A command to be executed
///
/// Call the following in order:
/// * If the concrete type uses a shader, then it will provide
///   the `ENTRY_POINT` constant and the
///   `bind_group_layout` associated method.
///   Get and cache the layout during startup. You'll need
///   these to call `[prepare]` and `[add_pass]`.
/// * `[prepare]`
/// * `[add_pass]`
/// * `[add_copy]`. This may be on a different queue, but the
///   copy's execution must happen after the pass's execution.
/// * `[async_finish]`. Only call this after the pass and copy
///   operations have finished executing on the GPU.
pub trait Command {
    /// Create buffers and bind group. get_bind_group_layout's argument
    /// is `ENTRY_POINT`.
    fn prepare<'a>(
        &mut self,
        device: &Device,
        get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    );

    /// Add the compute pass to the command encoder. get_pipeline's argument
    /// is `ENTRY_POINT`.
    fn add_pass<'a>(
        &self,
        encoder: &mut CommandEncoder,
        get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    );

    /// Add buffer copies, if any, to the command encoder
    fn add_copy(&self, encoder: &mut CommandEncoder);

    /// Map the copy buffers if needed and perform any finalization steps, then call the callback
    fn async_finish(&mut self, done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>);
}

fn _verify_object_safety() {
    let _: &dyn Command = &CreateGridCommand::default();
}

/// Create a voxel grid with the given size.
#[derive(Clone, Debug, Default)]
pub struct CreateGridCommand {
    /// Destination. Reuse the existing buffer without clearing if it already exists
    /// and its size matches.
    grid: SharedVoxelGridBuffer,

    /// Size of the voxel grid, excluding padding
    size: UVec3,
}

impl Command for CreateGridCommand {
    fn prepare<'a>(
        &mut self,
        device: &Device,
        _get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    ) {
        let mut guard = self.grid.lock();
        if let Some(grid) = &*guard {
            if grid.size == self.size {
                return;
            }
        }
        *guard = Some(VoxelGridBuffer::new(self.size, device, false));
    }

    fn add_pass<'a>(
        &self,
        _encoder: &mut CommandEncoder,
        _get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    ) {
    }

    fn add_copy(&self, _encoder: &mut CommandEncoder) {}

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        done(Ok(()));
    }
} // impl Command for CreateGridCommand

/// Convert a voxel grid to a mesh.
pub struct GenerateMeshCommand {
    /// Grid to turn into a mesh
    pub grid: SharedVoxelGridBuffer,

    /// Receives the generated vertexes and normals
    pub receive_result: Arc<dyn Fn(Vec<Vec3>, Vec<Vec3>) + 'static + Sync + Send>,

    cmd_impl: Option<GenerateMeshImpl>,
}

impl GenerateMeshCommand {
    /// Shader entry point
    pub const ENTRY_POINT: &'static str = GENERATE_MESH_ENTRY_POINT;

    /// Create bind group layout
    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        generate_mesh_bind_group_layout(device)
    }

    pub fn new(
        grid: SharedVoxelGridBuffer,
        receive_result: Arc<dyn Fn(Vec<Vec3>, Vec<Vec3>) + 'static + Sync + Send>,
    ) -> Self {
        Self {
            grid,
            receive_result,
            cmd_impl: Default::default(),
        }
    }
}

impl Command for GenerateMeshCommand {
    fn prepare<'a>(
        &mut self,
        device: &Device,
        get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    ) {
        let guard = self.grid.lock();
        self.cmd_impl = Some(GenerateMeshImpl::new(
            device,
            get_bind_group_layout(Self::ENTRY_POINT),
            guard.as_ref().expect("Missing grid in GenerateMeshCommand"),
        ));
    }

    fn add_pass<'a>(
        &self,
        encoder: &mut CommandEncoder,
        get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    ) {
        self.cmd_impl
            .as_ref()
            .unwrap()
            .add_pass(get_pipeline(Self::ENTRY_POINT), encoder);
    }

    fn add_copy(&self, encoder: &mut CommandEncoder) {
        self.cmd_impl.as_ref().unwrap().add_copy(encoder);
    }

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        let receive_result = self.receive_result.clone();
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
} // impl Command for GenerateMeshCommand

/// Type of geometry operation to perform
#[derive(Debug, Clone)]
pub enum GeometryOp {
    Sphere {
        /// Diameter of sphere
        diameter: u32,

        /// Offset sphere's coordinates
        offset: IVec3,

        /// Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
        /// Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
        flags: u32,

        /// Material to paste
        material: u32,
    },
}

/// Apply geometry to a mesh
#[derive(Debug)]
pub struct GeometryCommand {
    /// Grid to operate on
    pub grid: SharedVoxelGridBuffer,

    /// Type of geometry operation to perform
    pub geometry: GeometryOp,

    cmd_impl: Option<GeometryImpl>,
}

impl GeometryCommand {
    /// PasteSphere entry point
    pub const PASTE_SPHERE_ENTRY_POINT: &'static str = PASTE_SPHERE_ENTRY_POINT;

    /// Create bind group layout. This is the same for all geometry operations.
    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        geometry_bind_group_layout(device)
    }

    /// Create the command
    pub fn new(grid: SharedVoxelGridBuffer, geometry: GeometryOp) -> Self {
        Self {
            grid,
            geometry,
            cmd_impl: Default::default(),
        }
    }
}

impl Command for GeometryCommand {
    fn prepare<'a>(
        &mut self,
        device: &Device,
        get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    ) {
        let guard = self.grid.lock();
        let grid = guard.as_ref().expect("Missing grid in GeometryCommand");
        match &self.geometry {
            GeometryOp::Sphere {
                diameter,
                offset,
                flags,
                material,
            } => {
                self.cmd_impl = Some(GeometryImpl::new_sphere(
                    device,
                    get_bind_group_layout(Self::PASTE_SPHERE_ENTRY_POINT),
                    grid,
                    *diameter,
                    *offset,
                    *flags,
                    *material,
                ));
            }
        }
    }

    fn add_pass<'a>(
        &self,
        encoder: &mut CommandEncoder,
        get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    ) {
        let entry_point = match &self.geometry {
            GeometryOp::Sphere { .. } => Self::PASTE_SPHERE_ENTRY_POINT,
        };
        self.cmd_impl
            .as_ref()
            .unwrap()
            .add_pass(get_pipeline(entry_point), encoder);
    }

    fn add_copy(&self, _encoder: &mut CommandEncoder) {}

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        done(Ok(()));
    }
} // impl Command for GenerateMeshCommand

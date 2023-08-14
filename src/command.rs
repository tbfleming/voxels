use bytemuck::cast_slice;
use glam::{IVec3, UVec3, Vec3};
use parking_lot::Mutex;
use std::{
    fmt::Debug,
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use wgpu::{
    BindGroupLayout, Buffer, BufferAsyncError, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipeline, Device, MapMode,
};

use crate::voxel::*;

// lock order: SharedVoxelGridContent, SharedVoxelGrid
#[derive(Debug, Clone, Default)]
pub struct SharedVoxelGrid(Arc<Mutex<Option<VoxelGrid>>>);

impl SharedVoxelGrid {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Deref for SharedVoxelGrid {
    type Target = Arc<Mutex<Option<VoxelGrid>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SharedVoxelGrid {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
pub trait VoxelCommand {
    /// Create a boxed version of this command suitable for `[VoxelCommandVec]`.
    fn boxed(self) -> Box<dyn VoxelCommand + Send + Sync>
    where
        Self: 'static + Sized + Send + Sync,
    {
        Box::new(self)
    }

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

pub type VoxelCommandVec = Vec<Box<dyn VoxelCommand + Send + Sync>>;

/// Create a voxel grid with the given size.
#[derive(Clone, Debug, Default)]
pub struct CreateGridCommand {
    /// Destination. Reuse the existing buffer without clearing if it already exists
    /// and its size matches.
    grid: SharedVoxelGrid,

    /// Size of the voxel grid, excluding padding
    size: UVec3,
}

impl CreateGridCommand {
    pub fn new(grid: SharedVoxelGrid, size: UVec3) -> Self {
        Self { grid, size }
    }
}

impl VoxelCommand for CreateGridCommand {
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
        // println!("** Creating grid: {:?}", self.size);
        *guard = Some(VoxelGrid::new(self.size, device, false));
    }

    fn add_pass<'a>(
        &self,
        _encoder: &mut CommandEncoder,
        _get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    ) {
    }

    fn add_copy(&self, _encoder: &mut CommandEncoder) {}

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        // println!("@@@ CreateGridCommand::async_finish: nop");
        done(Ok(()));
    }
} // impl Command for CreateGridCommand

#[derive(Clone)]
/// Create a voxel grid with the given size.
pub struct GetVoxelsCommand {
    // Retrieve voxels from this grid
    grid: SharedVoxelGrid,

    // Receives result
    callback: Arc<dyn Fn(VoxelGridVec) + Send + Sync>,

    // Size of grid at the time it gets copied
    size: UVec3,

    // Size of the buffer to copy
    buffer_size: usize,

    // Retrieves the content. COPY_DST | MAP_READ
    copy_buffer: Arc<Mutex<Option<Buffer>>>,
}

impl GetVoxelsCommand {
    pub fn new(grid: SharedVoxelGrid, callback: Arc<dyn Fn(VoxelGridVec) + Send + Sync>) -> Self {
        Self {
            grid,
            callback,
            size: Default::default(),
            buffer_size: Default::default(),
            copy_buffer: Default::default(),
        }
    }
}

impl VoxelCommand for GetVoxelsCommand {
    fn prepare<'a>(
        &mut self,
        device: &Device,
        _get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    ) {
        let guard = self.grid.lock();
        let Some(grid) = &*guard else { return };
        self.size = grid.size;
        self.buffer_size = get_buf_size(grid.size);
        *self.copy_buffer.lock() = Some(device.create_buffer(&BufferDescriptor {
            label: None,
            size: self.buffer_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }

    fn add_pass<'a>(
        &self,
        _encoder: &mut CommandEncoder,
        _get_pipeline: &mut dyn FnMut(&str) -> &'a ComputePipeline,
    ) {
    }

    fn add_copy(&self, encoder: &mut CommandEncoder) {
        let src = self.grid.lock();
        let dest = self.copy_buffer.lock();
        encoder.copy_buffer_to_buffer(
            &src.as_ref().unwrap().buffer,
            0,
            dest.as_ref().unwrap(),
            0,
            self.buffer_size as u64,
        );
    }

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        let callback = self.callback.clone();
        let size = self.size;
        let copy_buffer = self.copy_buffer.clone();
        // println!("@@@ GetVoxelsCommand::async_finish mapping...");
        self.copy_buffer
            .lock()
            .as_ref()
            .unwrap()
            .slice(..)
            .map_async(MapMode::Read, move |result| {
                // println!("@@@ GetVoxelsCommand::async_finish mapped: {:?}", result);
                if result.is_ok() {
                    let guard = copy_buffer.lock();
                    let raw = guard.as_ref().unwrap().slice(..).get_mapped_range();
                    let mut data = Vec::new();
                    data.resize(raw.len() / size_of::<u32>(), 0);
                    data.copy_from_slice(cast_slice::<u8, u32>(&raw));
                    callback(VoxelGridVec { size, data });
                }
                done(result);
            });
    }
} // impl VoxelCommand for GetVoxelsCommand

/// Convert a voxel grid to a mesh.
pub struct GenerateMeshCommand {
    /// Grid to turn into a mesh
    pub grid: SharedVoxelGrid,

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
        grid: SharedVoxelGrid,
        receive_result: Arc<dyn Fn(Vec<Vec3>, Vec<Vec3>) + 'static + Sync + Send>,
    ) -> Self {
        Self {
            grid,
            receive_result,
            cmd_impl: Default::default(),
        }
    }
}

impl VoxelCommand for GenerateMeshCommand {
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
        // println!("@@@ GenerateMeshCommand::async_finish mapping...");
        self.cmd_impl
            .take()
            .unwrap()
            .async_map_buffer(move |cmd_impl, res| {
                // println!("@@@ GenerateMeshCommand::async_finish mapped: {:?}", res);
                if res.is_ok() {
                    let (m, n) = cmd_impl.get_mesh();
                    receive_result(m, n);
                }
                done(res);
            });
    }
} // impl Command for GenerateMeshCommand

/// Type of geometry operation to perform
#[derive(Debug, Clone)]
pub enum GeometryOp {
    PasteCube {
        /// Size of cube
        size: UVec3,

        /// Offset cube's coordinates
        offset: IVec3,

        /// Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
        /// Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
        flags: u32,

        /// Material to paste
        material: u32,
    },

    PasteSphere {
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
    pub grid: SharedVoxelGrid,

    /// Type of geometry operation to perform
    pub geometry: GeometryOp,

    cmd_impl: Option<GeometryImpl>,
}

impl GeometryCommand {
    /// Shader entry point
    pub const PASTE_CUBE_ENTRY_POINT: &'static str = PASTE_CUBE_ENTRY_POINT;

    /// Shader entry point
    pub const PASTE_SPHERE_ENTRY_POINT: &'static str = PASTE_SPHERE_ENTRY_POINT;

    /// Create bind group layout. This is the same for all geometry operations.
    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        geometry_bind_group_layout(device)
    }

    /// Create a command
    pub fn new(grid: SharedVoxelGrid, geometry: GeometryOp) -> Self {
        // println!("@@@ GeometryCommand::new");
        Self {
            grid,
            geometry,
            cmd_impl: None,
        }
    }

    /// Create a cube command
    pub fn cube(
        grid: SharedVoxelGrid,
        size: UVec3,
        offset: IVec3,
        flags: u32,
        material: u32,
    ) -> Self {
        Self::new(
            grid,
            GeometryOp::PasteCube {
                size,
                offset,
                flags,
                material,
            },
        )
    }

    /// Create a sphere command
    pub fn sphere(
        grid: SharedVoxelGrid,
        diameter: u32,
        offset: IVec3,
        flags: u32,
        material: u32,
    ) -> Self {
        Self::new(
            grid,
            GeometryOp::PasteSphere {
                diameter,
                offset,
                flags,
                material,
            },
        )
    }
}

impl VoxelCommand for GeometryCommand {
    fn prepare<'a>(
        &mut self,
        device: &Device,
        get_bind_group_layout: &mut dyn FnMut(&str) -> &'a BindGroupLayout,
    ) {
        // println!("@@@ GeometryCommand::prepare");
        let guard = self.grid.lock();
        let grid = guard.as_ref().expect("Missing grid in GeometryCommand");
        match &self.geometry {
            GeometryOp::PasteCube {
                size,
                offset,
                flags,
                material,
            } => {
                // println!("@@@ GeometryCommand::prepare: PasteCube");
                self.cmd_impl = Some(GeometryImpl::paste_cube(
                    device,
                    get_bind_group_layout(Self::PASTE_CUBE_ENTRY_POINT),
                    grid,
                    *size,
                    *offset,
                    *flags,
                    *material,
                ));
            }

            GeometryOp::PasteSphere {
                diameter,
                offset,
                flags,
                material,
            } => {
                // println!(
                //     "@@@ GeometryCommand::prepare: PasteSphere: diameter: {}",
                //     diameter
                // );
                self.cmd_impl = Some(GeometryImpl::paste_sphere(
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
        // println!("@@@ GeometryCommand::add_pass");
        let entry_point = match &self.geometry {
            GeometryOp::PasteCube { .. } => Self::PASTE_CUBE_ENTRY_POINT,
            GeometryOp::PasteSphere { .. } => Self::PASTE_SPHERE_ENTRY_POINT,
        };
        self.cmd_impl
            .as_ref()
            .unwrap()
            .add_pass(get_pipeline(entry_point), encoder);
    }

    fn add_copy(&self, _encoder: &mut CommandEncoder) {}

    fn async_finish(&mut self, mut done: Box<dyn FnMut(Result<(), BufferAsyncError>) + Send>) {
        // println!("@@@ GeometryCommand::async_finish: nop");
        done(Ok(()));
    }
} // impl Command for GeometryCommand

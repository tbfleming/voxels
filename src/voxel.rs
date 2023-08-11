use bytemuck::cast_slice;
use glam::{UVec3, Vec4};
use std::mem::size_of;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

pub(crate) const WGSL_VOXEL_GRID_IN_SIZE_BINDING: u32 = 2;
pub(crate) const WGSL_VOXEL_GRID_IN_BINDING: u32 = 3;
pub(crate) const WGSL_MESH_BINDING: u32 = 0;
pub(crate) const WGSL_FACE_FILLED_BINDING: u32 = 1;

pub(crate) const WGSL_VEC3_STRIDE: usize = size_of::<Vec4>(); // WGSL pads vec3
pub(crate) const WGSL_FACE_STRIDE: usize = WGSL_VEC3_STRIDE * VERTEXES_PER_FACE;
pub(crate) const WGSL_FACES_STRIDE: usize = WGSL_FACE_STRIDE * FACES_PER_VOXEL;

pub(crate) const VERTEXES_PER_FACE: usize = 6;
pub(crate) const FACES_PER_VOXEL: usize = 6;
pub(crate) const FACE_FILLED_NUM_BITS: u32 = 30;
pub(crate) const GENERATE_MESH_WORKGROUP_SIZE: u32 = 64;
pub(crate) const GENERATE_MESH_VOXELS_PER_INVOCATION: u32 = 5;
pub(crate) const GENERATE_MESH_VOXELS_PER_WORKGROUP: u32 =
    GENERATE_MESH_VOXELS_PER_INVOCATION * GENERATE_MESH_WORKGROUP_SIZE;

/// Voxels stored in a [Vec].
///
/// Each voxel is 4 bytes:
/// * Byte 0: `i8:` `offset_x * 64`
/// * Byte 1: `i8:` `offset_y * 64`
/// * Byte 2: `i8:` `offset_z * 64`
/// * Byte 3: `u8:` material. 0 means empty.
///
/// The offsets modify the voxel's lower-left corner and have range (-2.0, 2.0),
/// where 1.0 is the distance between voxel centers. If an offset value is 0x80,
/// it is treated as 0x81.
///
/// The voxels are packed by x, then y, then z. Each dimension is padded on both
/// sides by 1 voxel. The offsets at the start padding don't matter. The
/// offsets at the end padding complete the voxel bounds. Non-0 material in padding
/// excludes the faces at the edges of the voxel grid.
///
/// `index = (x + 1) + (y + 1) * (size.x + 2) + (z + 1) * (size.x + 2) * (size.y + 2)`,
/// where `0,0,0` is the lower-left voxel, skipping padding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VoxelGridVec {
    /// Size of the voxel grid, excluding padding
    pub size: UVec3,

    /// Voxel data, including padding
    pub data: Vec<u32>,
}

impl VoxelGridVec {
    /// Create a new voxel grid with the given size and material.
    /// The size does not include padding, but the result includes
    /// it. The Padding is filled with empty voxels.
    ///
    /// Panics if the size is too large.
    pub fn new(size: UVec3, material: u8) -> Self {
        let mut data = Vec::new();
        data.resize(get_vec_size(size), 0);
        if material != 0 {
            for z in 0..size.z {
                for y in 0..size.y {
                    for x in 0..size.x {
                        data[voxel_index(size, x, y, z)] = (material as u32) << 24;
                    }
                }
            }
        }
        Self { size, data }
    }
}

fn check_grid_size(size: UVec3) -> (usize, usize) {
    if size.x >= (i32::MAX - 2) as u32
        || size.y >= (i32::MAX - 2) as u32
        || size.z >= (i32::MAX - 2) as u32
    {
        panic!("Voxel grid size is too large");
    }
    let vec_size = (size.x as usize + 2) * (size.y as usize + 2) * (size.z as usize + 2);
    let buf_size = vec_size * size_of::<u32>();
    if buf_size >= i32::MAX as usize {
        panic!("Voxel grid size is too large");
    }
    (vec_size, buf_size)
}

/// Get the length of the data vector for a voxel grid with the given size.
/// The size does not include padding, but the returned value does.
///
/// Panics if the size is too large.
pub fn get_vec_size(size: UVec3) -> usize {
    check_grid_size(size).0
}

/// Get the length of the gpu buffer, in bytes, for a voxel grid with the given size.
/// The size does not include padding, but the returned value does.
///
/// Panics if the size is too large.
pub fn get_buf_size(size: UVec3) -> usize {
    check_grid_size(size).1
}

/// Get the index of a voxel in the data vector. `0,0,0` gets the first voxel,
/// skipping the padding. `size.<c>` for coordinate `c` (x, y, or z) gets ending padding.
///
/// This function doesn't check for out-of-bounds coordinates.
pub fn voxel_index(size: UVec3, x: u32, y: u32, z: u32) -> usize {
    ((x + 1) + (y + 1) * (size.x + 2) + (z + 1) * (size.x + 2) * (size.y + 2)) as usize
}

/// Get the index of a voxel in the data vector. `0,0,0` gets the first voxel,
/// skipping the padding. -1 for any coordinate gets beginning padding.
/// `size.<c>` for coordinate `c` (x, y, or z) gets ending padding.
///
/// This function doesn't check for out-of-bounds coordinates.
pub fn voxel_index_i32(size: UVec3, x: i32, y: i32, z: i32) -> usize {
    ((x + 1) + (y + 1) * (size.x as i32 + 2) + (z + 1) * (size.x as i32 + 2) * (size.y as i32 + 2))
        as usize
}

/// Voxels readable and writable by the GPU. See [VoxelGridContent] for the format.
#[derive(Debug)]
pub struct VoxelGridBuffer {
    /// Size of the voxel grid, excluding padding
    pub size: UVec3,

    /// Voxel data, including padding. Usage flags are
    /// `[BufferUsages::STORAGE] | [BufferUsages::COPY_SRC]`.
    pub buffer: Buffer,
}

impl VoxelGridBuffer {
    /// Create a new voxel grid with the given size. The size does
    /// not include padding, but the result includes it.
    ///
    /// Panics if the size is too large.
    pub fn new(size: UVec3, device: &Device, mapped_at_creation: bool) -> Self {
        Self {
            size,
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some("voxel_grid_buffer"),
                size: get_buf_size(size) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation,
            }),
        }
    }

    /// Create a new voxel grid and copy the given content into it.
    pub fn from_content(content: &VoxelGridVec, device: &Device) -> Self {
        let buffer = Self::new(content.size, device, true);
        buffer
            .buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice::<u32, u8>(&content.data));
        buffer.buffer.unmap();
        buffer
    }
}

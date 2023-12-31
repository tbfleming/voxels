use bytemuck::{cast_slice, checked::from_bytes_mut};
use glam::{IVec3, UVec3, Vec3, Vec4};
use std::{mem::size_of, num::NonZeroU64, sync::Arc};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAsyncError, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePipeline, Device, MapMode, ShaderStages,
};

pub const GENERATE_MESH_ENTRY_POINT: &str = "generate_mesh";
pub const PASTE_CUBE_ENTRY_POINT: &str = "paste_cube";
pub const PASTE_SPHERE_ENTRY_POINT: &str = "paste_sphere";

pub mod unstable {
    use bytemuck::{Pod, Zeroable};
    use glam::IVec3;

    use super::*;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
    pub struct ShaderArgs {
        pub a_size: UVec3,
        pub _0: u32, // padding
        pub b_size: UVec3,
        pub _1: u32,
        pub out_size: UVec3,
        pub _2: u32,
        pub offset: IVec3,
        pub _3: u32,
        pub size: UVec3,
        pub flags: u32,
        pub material: u32,
        pub diameter: u32,
        pub _4: u32,
        pub _5: u32,
    }

    pub const WGSL_ARGS_BINDING: u32 = 0;
    pub const WGSL_VOXEL_GRID_A_BINDING: u32 = 1;
    pub const WGSL_VOXEL_GRID_B_BINDING: u32 = 2;
    pub const WGSL_VOXEL_GRID_OUT_BINDING: u32 = 3;
    pub const WGSL_FACE_FILLED_BINDING: u32 = 4;
    pub const WGSL_MESH_BINDING: u32 = 5;
    pub const WGSL_MESH_NORMALS_BINDING: u32 = 6;

    pub const WGSL_VEC3_STRIDE: usize = size_of::<Vec4>(); // WGSL pads vec3
    pub const WGSL_FACE_STRIDE: usize = WGSL_VEC3_STRIDE * VERTEXES_PER_FACE;
    pub const WGSL_FACES_STRIDE: usize = WGSL_FACE_STRIDE * FACES_PER_VOXEL;

    pub const VERTEXES_PER_FACE: usize = 6;
    pub const FACES_PER_VOXEL: usize = 6;
    pub const FACE_FILLED_NUM_BITS: u32 = 30;
    pub const GENERATE_MESH_WORKGROUP_SIZE: u32 = 64;
    pub const GENERATE_MESH_VOXELS_PER_INVOCATION: u32 = 5;
    pub const GENERATE_MESH_VOXELS_PER_WORKGROUP: u32 =
        GENERATE_MESH_VOXELS_PER_INVOCATION * GENERATE_MESH_WORKGROUP_SIZE;

    pub const PASTE_CUBE_VOXELS_PER_WORKGROUP: u32 = 64;
    pub const PASTE_SPHERE_VOXELS_PER_WORKGROUP: u32 = 64;
}

use unstable::*;

pub const PASTE_MATERIAL_FLAG: u32 = 1;
pub const PASTE_MATERIAL_ARG_FLAG: u32 = 2;
pub const PASTE_VERTEXES_FLAG: u32 = 4;
pub const PASTE: u32 = PASTE_MATERIAL_FLAG | PASTE_VERTEXES_FLAG;

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
pub struct VoxelGrid {
    /// Size of the voxel grid, excluding padding
    pub size: UVec3,

    /// Voxel data, including padding. Usage flags are
    /// `[BufferUsages::STORAGE] | [BufferUsages::COPY_SRC]`.
    pub buffer: Buffer,
}

impl VoxelGrid {
    /// Create a new voxel grid with the given size. The size does
    /// not include padding, but the result includes it.
    ///
    /// Panics if the size is too large.
    pub fn new(size: UVec3, device: &Device, mapped_at_creation: bool) -> Self {
        // println!(
        //     "** VoxelGrid::new {} {} {}",
        //     size,
        //     get_buf_size(size),
        //     mapped_at_creation
        // );
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("voxel_grid_buffer"),
            size: get_buf_size(size) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation,
        });
        // println!("** buffer {} {:?}", buffer.size(), buffer.usage());
        Self { size, buffer }
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

/// Create BindGroupLayout for the shader's generate_mesh function.
pub fn generate_mesh_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("generate_mesh_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: WGSL_ARGS_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: WGSL_VOXEL_GRID_A_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: WGSL_FACE_FILLED_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: WGSL_MESH_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: WGSL_MESH_NORMALS_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Use the the shader's generate_mesh function to convert a
/// voxel grid in VoxelGridBuffer to a mesh.
///
/// Call the following in order:
/// * `[new]`
/// * `[add_pass]`
/// * `[add_copy]`. This may be on a different queue, but the
///   copy's execution must happen after the pass's execution.
/// * `[async_map_buffer]`. Only call this after the copy has
///   finished executing on the GPU.
/// * `[get_mesh]`. Only call this after async_map_buffer has
///   called its callback.
#[derive(Debug)]
pub struct GenerateMeshImpl {
    // Excludes padding
    num_voxels: usize,

    // Offset of normals in storage_buffer
    normals_offset: usize,

    // Offset of face_filled in storage_buffer
    face_filled_offset: usize,

    // Size of storage_buffer and copy_buffer
    buffer_size: usize,

    // Receives the raw mesh from the shader. STORAGE | COPY_SRC
    storage_buffer: Buffer,

    // Copy of storage_buffer. COPY_DST | MAP_READ
    copy_buffer: Arc<Buffer>,

    bind_group: BindGroup,
}

pub fn vec4_to_3(v: &Vec4) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

impl GenerateMeshImpl {
    /// Create buffers and bind group
    pub fn new(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        grid_buffer: &VoxelGrid,
    ) -> Self {
        // println!("** GenerateMeshImpl::new");
        let num_voxels =
            grid_buffer.size.x as usize * grid_buffer.size.y as usize * grid_buffer.size.z as usize;
        // println!("   num_voxels: {:?}", num_voxels);
        let normals_offset = num_voxels * WGSL_FACES_STRIDE;
        let face_filled_offset = normals_offset + num_voxels * WGSL_FACES_STRIDE;
        // println!("   face_filled_offset: {:?}", face_filled_offset);
        let num_faces = num_voxels * FACES_PER_VOXEL;
        let buffer_size = face_filled_offset
            + (num_faces + FACE_FILLED_NUM_BITS as usize - 1) / FACE_FILLED_NUM_BITS as usize * 4;
        // println!(
        //     "    grid_buffer {} {:?}",
        //     grid_buffer.buffer.size(),
        //     grid_buffer.buffer.usage()
        // );

        let args = ShaderArgs {
            a_size: grid_buffer.size,
            ..Default::default()
        };
        let args_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<ShaderArgs>() as u64,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        *from_bytes_mut::<ShaderArgs>(&mut args_buffer.slice(..).get_mapped_range_mut()) = args;
        args_buffer.unmap();

        let storage_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let copy_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("generate_mesh_bind_group"),
            layout: bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: WGSL_ARGS_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &args_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_VOXEL_GRID_A_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &grid_buffer.buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_FACE_FILLED_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &storage_buffer,
                        offset: face_filled_offset as u64,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_MESH_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &storage_buffer,
                        offset: 0,
                        size: NonZeroU64::new(normals_offset as u64),
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_MESH_NORMALS_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &storage_buffer,
                        offset: normals_offset as u64,
                        size: NonZeroU64::new((face_filled_offset - normals_offset) as u64),
                    }),
                },
            ],
        });

        Self {
            num_voxels,
            normals_offset,
            face_filled_offset,
            buffer_size,
            storage_buffer,
            copy_buffer: copy_buffer.into(),
            bind_group,
        }
    }

    /// Add the compute pass to the command encoder
    pub fn add_pass(&self, pipeline: &ComputePipeline, encoder: &mut CommandEncoder) {
        // println!("** GenerateMeshImpl::add_pass");
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("generate_mesh_pass"),
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(
            (self.num_voxels as u32 + GENERATE_MESH_VOXELS_PER_WORKGROUP - 1)
                / GENERATE_MESH_VOXELS_PER_WORKGROUP,
            1,
            1,
        );
    }

    /// Add the buffer copy to the command encoder
    pub fn add_copy(&self, encoder: &mut CommandEncoder) {
        // println!("** GenerateMeshImpl::add_copy");
        encoder.copy_buffer_to_buffer(
            &self.storage_buffer,
            0,
            &self.copy_buffer,
            0,
            self.buffer_size as u64,
        );
    }

    /// Map the copy buffer (async) then call the callback
    pub fn async_map_buffer(
        self,
        done: impl FnOnce(GenerateMeshImpl, Result<(), BufferAsyncError>) + Send + 'static,
    ) {
        self.copy_buffer
            .clone()
            .slice(..)
            .map_async(MapMode::Read, |result| done(self, result));
    }

    /// Get the mesh and normals from the copy buffer
    pub fn get_mesh(self) -> (Vec<Vec3>, Vec<Vec3>) {
        let raw = self.copy_buffer.slice(..).get_mapped_range();
        let src_vertexes = cast_slice::<u8, Vec4>(&raw[..self.normals_offset]);
        let src_normals =
            cast_slice::<u8, Vec4>(&raw[self.normals_offset..self.face_filled_offset]);
        let face_filled = cast_slice::<u8, u32>(&raw[self.face_filled_offset..]);

        let mut num_faces = 0;
        for mask in face_filled {
            // println!("   mask: {:#08x}", mask);
            num_faces += mask.count_ones() as usize;
        }

        let mut vertexes: Vec<Vec3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        vertexes.resize(num_faces * VERTEXES_PER_FACE, Default::default());
        normals.resize(num_faces * VERTEXES_PER_FACE, Default::default());

        let mut filled = 0;
        for i in 0..self.num_voxels * FACES_PER_VOXEL {
            if face_filled[i / FACE_FILLED_NUM_BITS as usize]
                & (1 << (i % FACE_FILLED_NUM_BITS as usize))
                != 0
            {
                // println!("   fill face: {:?}", i);
                for j in 0..VERTEXES_PER_FACE {
                    let v = src_vertexes[i * VERTEXES_PER_FACE + j];
                    vertexes[filled * VERTEXES_PER_FACE + j] = vec4_to_3(&v);

                    let n = src_normals[i * VERTEXES_PER_FACE + j];
                    normals[filled * VERTEXES_PER_FACE + j] = vec4_to_3(&n);
                }
                filled += 1;
            }
        }
        // println!("   filled: {:?}", filled);
        // println!("   num_faces: {:?}", num_faces);
        assert!(filled == num_faces);
        (vertexes, normals)
    }
} // GenerateMeshImpl

/// Create BindGroupLayout for the shader's geometry functions.
pub fn geometry_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("geometry_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: WGSL_ARGS_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: WGSL_VOXEL_GRID_OUT_BINDING,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Use one of the shader's geometry functions.
///
/// Call the following in order:
/// * `[new_*]`
/// * `[add_pass]`
#[derive(Debug)]
pub struct GeometryImpl {
    bind_group: BindGroup,
    workgroup_size: UVec3,
}

impl GeometryImpl {
    fn new_impl(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        bind_group_label: &'static str,
        grid_buffer: &VoxelGrid,
        args: ShaderArgs,
        workgroup_size: UVec3,
    ) -> Self {
        // println!("** GeometryImpl::new_impl");
        // println!("   {:?}", args);
        // println!(
        //     "    grid_buffer {} {:?}",
        //     grid_buffer.buffer.size(),
        //     grid_buffer.buffer.usage()
        // );
        let args_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<ShaderArgs>() as u64,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        *from_bytes_mut::<ShaderArgs>(&mut args_buffer.slice(..).get_mapped_range_mut()) = args;
        args_buffer.unmap();
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(bind_group_label),
            layout: bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: WGSL_ARGS_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &args_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_VOXEL_GRID_OUT_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &grid_buffer.buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        Self {
            bind_group,
            workgroup_size,
        }
    }

    /// Create buffers and bind group for the shader's paste_cube function.
    ///
    /// * grid_buffer:  Voxel grid to modify
    /// * size:         Diameter of cube
    /// * offset:       Offset cube's coordinates
    /// * flags:        Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
    ///                 Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
    /// * material:     Material to paste
    pub fn paste_cube(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        grid_buffer: &VoxelGrid,
        size: UVec3,
        offset: IVec3,
        flags: u32,
        material: u32,
    ) -> Self {
        let args = ShaderArgs {
            out_size: grid_buffer.size,
            size,
            offset,
            flags,
            material,
            ..Default::default()
        };
        let workgroup_size =
            ((size.x + 1) * (size.y + 1) * (size.z + 1) + PASTE_CUBE_VOXELS_PER_WORKGROUP - 1)
                / PASTE_CUBE_VOXELS_PER_WORKGROUP;
        Self::new_impl(
            device,
            bind_group_layout,
            "paste_cube_bind_group",
            grid_buffer,
            args,
            UVec3::new(workgroup_size, 1, 1),
        )
    }

    /// Create buffers and bind group for the shader's paste_sphere function.
    ///
    /// * grid_buffer:  Voxel grid to modify
    /// * diameter:     Diameter of sphere
    /// * offset:       Offset sphere's coordinates
    /// * flags:        Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
    ///                 Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
    /// * material:     Material to paste
    pub fn paste_sphere(
        device: &Device,
        bind_group_layout: &BindGroupLayout,
        grid_buffer: &VoxelGrid,
        diameter: u32,
        offset: IVec3,
        flags: u32,
        material: u32,
    ) -> Self {
        let args = ShaderArgs {
            out_size: grid_buffer.size,
            offset,
            flags,
            material,
            diameter,
            ..Default::default()
        };
        let workgroup_size = ((diameter + 1) * (diameter + 1) * (diameter + 1)
            + PASTE_SPHERE_VOXELS_PER_WORKGROUP
            - 1)
            / PASTE_SPHERE_VOXELS_PER_WORKGROUP;
        Self::new_impl(
            device,
            bind_group_layout,
            "paste_sphere_bind_group",
            grid_buffer,
            args,
            UVec3::new(workgroup_size, 1, 1),
        )
    }

    /// Add the compute pass to the command encoder
    pub fn add_pass(&self, pipeline: &ComputePipeline, encoder: &mut CommandEncoder) {
        // println!("** GeometryImpl::add_pass");
        // println!("   workgroup_size: {:?}", self.workgroup_size);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("geometry_pass"),
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(
            self.workgroup_size.x,
            self.workgroup_size.y,
            self.workgroup_size.z,
        );
    }
} // GeometryImpl

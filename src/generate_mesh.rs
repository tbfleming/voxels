use bevy::{
    core::cast_slice,
    math::Vec4Swizzles,
    prelude::*,
    reflect::TypePath,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    },
};
use bytemuck::cast_slice_mut;
use std::{
    borrow::Cow,
    mem::{replace, size_of, take},
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use crate::voxel::*;

const WGSL_MESH_BINDING: u32 = 0;
const WGSL_FACE_FILLED_BINDING: u32 = 1;

const WGSL_VEC3_STRIDE: usize = size_of::<Vec4>(); // WGSL pads vec3
const WGSL_FACE_STRIDE: usize = WGSL_VEC3_STRIDE * VERTEXES_PER_FACE;
const WGSL_FACES_STRIDE: usize = WGSL_FACE_STRIDE * FACES_PER_VOXEL;

const VERTEXES_PER_FACE: usize = 6;
const FACES_PER_VOXEL: usize = 6;
const FACE_FILLED_NUM_BITS: u32 = 30;
const GENERATE_MESH_WORKGROUP_SIZE: u32 = 64;
const GENERATE_MESH_VOXELS_PER_INVOCATION: u32 = 5;

pub struct GenerateMeshPlugin;

impl Plugin for GenerateMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<GenerateMesh>::default());
        app.add_systems(First, finalize_generate_mesh);

        // TODO: order copy_data_to_storage first
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, prepare_generate_mesh.in_set(RenderSet::Prepare));
        render_app.add_systems(Render, map_generate_mesh.in_set(RenderSet::Cleanup));

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("generate_mesh", GenerationNode);
        render_graph.add_node_edge(
            "generate_mesh",
            bevy::render::main_graph::node::CAMERA_DRIVER,
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GenerationPipeline>();
    }
}

// lock order: VoxelGridStorageBuffer, GenerateMesh
#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct GenerateMesh(SharedGenerateMeshState);

impl GenerateMesh {
    pub fn new() -> Self {
        default()
    }
}

type SharedGenerateMeshState = Arc<Mutex<GenerateMeshState>>;

#[derive(Default, Debug)]
enum GenerateMeshState {
    #[default]
    Init,
    Busy(GenerateMeshData),
    Mapping,
    Mapped(GenerateMeshData),
    Done,
}

#[derive(Debug)]
struct GenerateMeshData {
    num_voxels: usize,
    face_filled_offset: usize,
    buffer_size: usize,
    storage_buffer: Buffer,
    copy_buffer: Buffer,
    bind_group: BindGroup,
}

fn prepare_generate_mesh(
    render_device: Res<RenderDevice>,
    mut pipeline: ResMut<GenerationPipeline>,
    generate_meshes: Query<(&GenerateMesh, &VoxelGridStorageBuffer)>,
) {
    // println!("** prepare_generate_mesh");
    for (generate_mesh, voxel_grid_storage_buffer) in generate_meshes.iter() {
        // println!("** prepare_generate_mesh: ?");
        let grid_buffer_guard = voxel_grid_storage_buffer.buffer.lock().unwrap();
        let mut mesh_state_guard = generate_mesh.0.lock().unwrap();
        let Some(grid_buffer) = &*grid_buffer_guard else {
            continue;
        };
        let GenerateMeshState::Init = &*mesh_state_guard else {
            continue;
        };
        println!("** prepare_generate_mesh: Init");
        println!("   size: {:?}", voxel_grid_storage_buffer.size);

        let num_voxels = voxel_grid_storage_buffer.size.x as usize
            * voxel_grid_storage_buffer.size.y as usize
            * voxel_grid_storage_buffer.size.z as usize;
        println!("   num_voxels: {:?}", num_voxels);
        let face_filled_offset = num_voxels * WGSL_FACES_STRIDE;
        println!("   face_filled_offset: {:?}", face_filled_offset);
        let num_faces = num_voxels * FACES_PER_VOXEL;
        let buffer_size = face_filled_offset
            + (num_faces + FACE_FILLED_NUM_BITS as usize - 1) / FACE_FILLED_NUM_BITS as usize * 4;
        println!("   buffer_size: {:?}", buffer_size);
        let storage_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let copy_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<Vec3>() as u64,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        cast_slice_mut::<u8, UVec3>(&mut uniform_buffer.slice(..).get_mapped_range_mut())[0] =
            voxel_grid_storage_buffer.size;
        uniform_buffer.unmap();

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: WGSL_MESH_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &storage_buffer,
                        offset: 0,
                        size: NonZeroU64::new(face_filled_offset as u64),
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
                    binding: WGSL_VOXEL_GRID_IN_SIZE_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: WGSL_VOXEL_GRID_IN_BINDING,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: grid_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        *mesh_state_guard = GenerateMeshState::Busy(GenerateMeshData {
            num_voxels,
            face_filled_offset,
            buffer_size,
            storage_buffer,
            copy_buffer,
            bind_group,
        });
        pipeline.states.push(generate_mesh.0.clone());
    }
}

fn map_generate_mesh(mut pipeline: ResMut<GenerationPipeline>) {
    for shared_state in take(&mut pipeline.states) {
        let mut guard = shared_state.lock().unwrap();
        let state = replace(&mut *guard, GenerateMeshState::Mapping);
        let GenerateMeshState::Busy(data) = state else {
            *guard = state;
            continue;
        };
        drop(guard);
        data.copy_buffer
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |res| {
                println!("mapped?: {:?}", res);
                if res.is_ok() {
                    *shared_state.lock().unwrap() = GenerateMeshState::Mapped(data);
                }
            });
    }
}

fn finalize_generate_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<(Entity, &GenerateMesh)>,
) {
    for (entity, generate_mesh) in query.iter_mut() {
        let mut guard = generate_mesh.0.lock().unwrap();
        {
            let GenerateMeshState::Mapped(data) = &*guard else {
                continue;
            };
            println!("** finalize_generate_mesh");

            let raw = data.copy_buffer.slice(..).get_mapped_range();
            let src_vertexes = cast_slice::<u8, Vec4>(&raw[..data.face_filled_offset]);
            let face_filled = cast_slice::<u8, u32>(&raw[data.face_filled_offset..]);

            let mut num_faces = 0;
            for mask in face_filled {
                num_faces += mask.count_ones() as usize;
            }

            let mut vertexes: Vec<Vec3> = Vec::new();
            vertexes.resize(num_faces * VERTEXES_PER_FACE, default());

            let mut filled = 0;
            for i in 0..data.num_voxels * FACES_PER_VOXEL {
                if face_filled[i / FACE_FILLED_NUM_BITS as usize]
                    & (1 << (i % FACE_FILLED_NUM_BITS as usize))
                    != 0
                {
                    for j in 0..VERTEXES_PER_FACE {
                        vertexes[filled * VERTEXES_PER_FACE + j] =
                            src_vertexes[i * VERTEXES_PER_FACE + j].xyz();
                    }
                    filled += 1;
                }
            }
            assert!(filled == num_faces);

            // println!("{:?}\n", src_vertexes);
            // println!("{:?}", vertexes);
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertexes);
            commands.entity(entity).insert(meshes.add(mesh));
        }
        *guard = GenerateMeshState::Done;
    }
}

#[derive(Resource)]
pub struct GenerationPipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    states: Vec<SharedGenerateMeshState>,
}

impl FromWorld for GenerationPipeline {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
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
                            binding: WGSL_VOXEL_GRID_IN_SIZE_BINDING,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: WGSL_VOXEL_GRID_IN_BINDING,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let shader = world.resource::<AssetServer>().load("shaders/vox.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("generate_mesh"),
        });

        GenerationPipeline {
            bind_group_layout,
            pipeline,
            states: Vec::new(),
        }
    }
}

#[derive(Default)]
struct GenerationNode;

impl render_graph::Node for GenerationNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GenerationPipeline>();
        for shared_state in pipeline.states.iter() {
            let guard = shared_state.lock().unwrap();
            let GenerateMeshState::Busy(data) = &*guard else {
                continue;
            };
            {
                let pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
                    .unwrap();
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_bind_group(0, &data.bind_group, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            render_context.command_encoder().copy_buffer_to_buffer(
                &data.storage_buffer,
                0,
                &data.copy_buffer,
                0,
                data.buffer_size as u64,
            );
        }

        Ok(())
    }
}

use bevy::{
    core::cast_slice,
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
use std::{
    borrow::Cow,
    mem::take,
    sync::{Arc, Mutex},
};

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        println!("** VoxelPlugin::build");
        app.add_plugins(ExtractComponentPlugin::<GeneratedMeshBuffer>::default());
        app.add_systems(First, generate_mesh);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_generate_mesh_buffers.in_set(RenderSet::Prepare),
        );
        render_app.add_systems(
            Render,
            cleanup_generate_mesh_buffers.in_set(RenderSet::Cleanup),
        );

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("foo", GenerationNode);
        render_graph.add_node_edge("foo", bevy::render::main_graph::node::CAMERA_DRIVER);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GenerationPipeline>();
    }
}

pub type SharedGeneratedMeshBuffer = Arc<Mutex<Option<Buffer>>>;

// Some(Buffer) is mapped with MapMode::Read
#[derive(Component, Deref, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct GeneratedMeshBuffer(pub SharedGeneratedMeshBuffer);

fn prepare_generate_mesh_buffers(
    render_device: Res<RenderDevice>,
    mut pipeline: ResMut<GenerationPipeline>,
    generated_meshes: Query<&GeneratedMeshBuffer>,
) {
    // println!("** prepare_generate_meshes");
    for generated_mesh in generated_meshes.iter() {
        // println!("** prepare_generate_meshes: ?");
        let guard = generated_mesh.lock().unwrap();
        if guard.is_none() {
            // println!("** prepare_generate_meshes: None");
            let storage_buffer = render_device.create_buffer(&BufferDescriptor {
                label: None,
                size: 432,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let copy_buffer = render_device.create_buffer(&BufferDescriptor {
                label: None,
                size: 432,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &storage_buffer,
                        offset: 0,
                        size: None,
                    }),
                }],
            });
            pipeline.data.push(GenerationData {
                generated_mesh: generated_mesh.0.clone(),
                storage_buffer,
                copy_buffer,
                bind_group,
            });
        }
    }
}

fn cleanup_generate_mesh_buffers(mut pipeline: ResMut<GenerationPipeline>) {
    for data in take(&mut pipeline.data) {
        data.copy_buffer
            .clone()
            .slice(..)
            .map_async(MapMode::Read, move |res| {
                println!("mapped?: {:?}", res);
                if res.is_ok() {
                    data.generated_mesh
                        .lock()
                        .unwrap()
                        .replace(data.copy_buffer);
                }
            });
    }
}

#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct GenerateMesh;

fn generate_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &GeneratedMeshBuffer), With<GenerateMesh>>,
) {
    for (entity, generated_mesh_buffer) in query.iter() {
        let mut guard = generated_mesh_buffer.lock().unwrap();
        if let Some(buffer) = guard.take() {
            println!("** generate_mesh");
            let src = buffer.slice(..).get_mapped_range();
            assert!(src.len() % 12 == 0);
            let mut dst: Vec<[f32; 3]> = Vec::new();
            dst.resize(src.len() / 12, [0.0, 0.0, 0.0]);
            dst.copy_from_slice(cast_slice::<u8, [f32; 3]>(&src));
            println!("{:?}", dst);

            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, dst);
            commands.entity(entity).insert(meshes.add(mesh));
        }
    }
}

struct GenerationData {
    generated_mesh: SharedGeneratedMeshBuffer,
    storage_buffer: Buffer,
    copy_buffer: Buffer,
    bind_group: BindGroup,
}

#[derive(Resource)]
pub struct GenerationPipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    data: Vec<GenerationData>,
}

impl FromWorld for GenerationPipeline {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let shader = world.resource::<AssetServer>().load("shaders/vox.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("foo"),
        });

        GenerationPipeline {
            bind_group_layout,
            pipeline,
            data: Vec::new(),
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
        for data in pipeline.data.iter() {
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
                432,
            );
        }

        Ok(())
    }
}

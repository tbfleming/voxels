use bevy::{
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
use parking_lot::Mutex;
use std::{
    borrow::Cow,
    mem::{replace, take},
    sync::Arc,
};

use crate::*;

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

#[derive(Component, Default, Clone, Debug, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct GenerateMesh(SharedGenerateMeshState);

impl GenerateMesh {
    pub fn new() -> Self {
        default()
    }
}

// lock order: SharedVoxelGridBuffer, GenerateMeshState
type SharedGenerateMeshState = Arc<Mutex<GenerateMeshState>>;

#[derive(Default, Debug)]
enum GenerateMeshState {
    #[default]
    Init,
    Busy(GenerateMeshImpl),
    Mapping,
    Mapped(GenerateMeshImpl),
    Done,
}

fn prepare_generate_mesh(
    render_device: Res<RenderDevice>,
    mut pipeline: ResMut<GenerationPipeline>,
    generate_meshes: Query<(&GenerateMesh, &VoxelGrid)>,
) {
    // println!("** prepare_generate_mesh");
    for (generate_mesh, voxel_grid) in generate_meshes.iter() {
        // println!("** prepare_generate_mesh: ?");
        let grid_buffer_guard = voxel_grid.lock();
        let mut mesh_state_guard = generate_mesh.0.lock();
        let Some(grid_buffer) = &*grid_buffer_guard else {
            continue;
        };
        let GenerateMeshState::Init = &*mesh_state_guard else {
            continue;
        };
        println!("** prepare_generate_mesh: Init");
        println!("   size: {:?}", grid_buffer.size);
        *mesh_state_guard = GenerateMeshState::Busy(GenerateMeshImpl::new(
            render_device.wgpu_device(),
            &pipeline.bind_group_layout,
            grid_buffer,
        ));
        pipeline.states.push(generate_mesh.0.clone());
    }
}

fn map_generate_mesh(mut pipeline: ResMut<GenerationPipeline>) {
    for shared_state in take(&mut pipeline.states) {
        let mut guard = shared_state.lock();
        let state = replace(&mut *guard, GenerateMeshState::Mapping);
        let GenerateMeshState::Busy(gen_impl) = state else {
            *guard = state;
            continue;
        };
        drop(guard);
        gen_impl.async_map_buffer(move |gen_impl, res| {
            println!("mapped?: {:?}", res);
            if res.is_ok() {
                *shared_state.lock() = GenerateMeshState::Mapped(gen_impl);
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
        let mut guard = generate_mesh.0.lock();
        let state = replace(&mut *guard, GenerateMeshState::Done);
        let GenerateMeshState::Mapped(gen_impl) = state else {
            *guard = state;
            continue;
        };
        println!("** finalize_generate_mesh");
        let (vertexes, normals) = gen_impl.get_mesh();
        // println!("{:?}\n", src_vertexes);
        // println!("{:?}", vertexes);
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertexes);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        commands.entity(entity).insert(meshes.add(mesh));
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
        let bind_group_layout: BindGroupLayout =
            generate_mesh_bind_group_layout(world.resource::<RenderDevice>().wgpu_device()).into();
        let shader = world.resource::<AssetServer>().load("shaders/vox.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("generate_mesh_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from(GENERATE_MESH_ENTRY_POINT),
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
            let guard = shared_state.lock();
            let GenerateMeshState::Busy(gen_impl) = &*guard else {
                continue;
            };
            let encoder = render_context.command_encoder();
            gen_impl.add_pass(
                pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
                    .unwrap(),
                encoder,
            );
            gen_impl.add_copy(encoder);
        }

        Ok(())
    }
}

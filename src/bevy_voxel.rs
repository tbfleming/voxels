use bevy::{
    ecs::query::Has,
    prelude::*,
    reflect::TypePath,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{self, RenderGraph},
        render_resource::{
            BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        },
        renderer::{RenderContext, RenderDevice},
        Render, RenderApp, RenderSet,
    },
};
use parking_lot::{Mutex, MutexGuard};
use std::{
    borrow::Cow,
    collections::HashMap,
    mem::take,
    ops::{Deref, DerefMut},
    sync::atomic::{self, AtomicUsize},
    sync::Arc,
};

use crate::command::*;
use crate::voxel::*;

pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<VoxelGridData>::default());
        app.add_plugins(ExtractComponentPlugin::<VoxelGrid>::default());
        app.add_plugins(ExtractComponentPlugin::<CopyDataToVoxelGrid>::default());
        app.add_plugins(ExtractComponentPlugin::<CommandList>::default());
        app.add_systems(First, finalize_copy_data_to_voxel_grid);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(Render, copy_data_to_voxel_grid.in_set(RenderSet::Prepare));
        render_app.add_systems(Render, prepare_command_list.in_set(RenderSet::Prepare));
        render_app.add_systems(Render, map_commands.in_set(RenderSet::Cleanup));

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("voxel_command_lists", VoxelCommandListsNode);
        render_graph.add_node_edge(
            "voxel_command_lists",
            bevy::render::main_graph::node::CAMERA_DRIVER,
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<CommandPipeline>();
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

// TODO: move this to the command system?
fn copy_data_to_voxel_grid(
    render_device: Res<RenderDevice>,
    query: Query<(&VoxelGridData, &VoxelGrid), Has<CopyDataToVoxelGrid>>,
) {
    for (voxel_grid_data, voxel_grid_storage_buffer) in query.iter() {
        let data_lock = voxel_grid_data.lock();
        let mut vg_buffer_lock = voxel_grid_storage_buffer.lock();
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

// TODO: move this to the command system?
fn finalize_copy_data_to_voxel_grid(
    mut commands: Commands,
    query: Query<(Entity, &VoxelGrid), Has<CopyDataToVoxelGrid>>,
) {
    for (entity, voxel_grid_storage_buffer) in query.iter() {
        let vg_buffer = voxel_grid_storage_buffer.lock();
        if vg_buffer.is_some() {
            commands.entity(entity).remove::<CopyDataToVoxelGrid>();
        }
    }
}

pub type CommandVec = Vec<Box<dyn Command + Send + Sync>>;

/// A list of commands that can be run on the GPU.
///
/// This acts as a handle; clones point to the same list.
#[derive(Component, Default, Clone, TypePath, ExtractComponent)]
#[component(storage = "SparseSet")]
pub struct CommandList(SharedCommandListData);

impl CommandList {
    /// Create a new command list.
    pub fn new(commands: CommandVec) -> Self {
        Self(Arc::new(CommandListData {
            state: CommandListState::Init.into(),
            commands: commands.into(),
        }))
    }

    /// Get mutable access to the commands. Returns None if the command list
    /// is not in the Init or Done state.
    ///
    /// This locks the list's mutex; the guard keeps it locked until dropped.
    pub fn commands_mut(&self) -> Option<CommandGuard<'_>> {
        let guard = self.0.lock();
        if *guard.state != CommandListState::Init || *guard.state == CommandListState::Done {
            Some(guard)
        } else {
            None
        }
    }

    /// Get the current state.
    ///
    /// This locks the list's Mutex.
    pub fn state(&self) -> CommandListState {
        *self.0.state.lock()
    }

    /// Switch the command list to the Init state and return true.
    /// Returns false if the command list is currently busy.
    ///
    /// This locks the list's mutex.
    pub fn run_again(&self) -> bool {
        let mut guard = self.0.state.lock();
        if *guard == CommandListState::Done {
            *guard = CommandListState::Init;
        }
        *guard == CommandListState::Init
    }
}

/// Mutable access to the commands in a command list. This keeps the list's
/// mutex locked until dropped.
pub struct CommandGuard<'a> {
    state: MutexGuard<'a, CommandListState>, // dropped first
    commands: MutexGuard<'a, CommandVec>,
}

impl<'a> Deref for CommandGuard<'a> {
    type Target = CommandVec;

    fn deref(&self) -> &Self::Target {
        &self.commands
    }
}

impl<'a> DerefMut for CommandGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.commands
    }
}

/// State the a command list can be in.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandListState {
    /// The command list is ready to be run.
    #[default]
    Init,

    /// The command list is currently being processed.
    Busy,

    /// The command list has buffers that are being mapped.
    Mapping,

    /// The command list is done.
    Done,
}

#[derive(Default)]
pub struct CommandListData {
    // lock order: commands, state
    state: Mutex<CommandListState>,
    commands: Mutex<CommandVec>,
}

impl CommandListData {
    fn lock(&self) -> CommandGuard {
        let commands = self.commands.lock();
        let state = self.state.lock();
        CommandGuard { state, commands }
    }
}

type SharedCommandListData = Arc<CommandListData>;

fn prepare_command_list(
    render_device: Res<RenderDevice>,
    mut pipeline: ResMut<CommandPipeline>,
    query: Query<&CommandList>,
) {
    // println!("** prepare_command_list");
    for command_list in query.iter() {
        // println!("** prepare_command_list: ?");
        let mut guard = command_list.0.lock();
        let CommandListState::Init = *guard.state else {
            continue;
        };
        println!("** prepare_command_list: Init");
        println!("   commands: {:?}", guard.commands.len());
        for command in guard.commands.iter_mut() {
            command.prepare(render_device.wgpu_device(), &mut |name| {
                if let Some(entry) = pipeline.map.get(name) {
                    &entry.layout
                } else {
                    panic!("Unknown bind group layout in commands: {}", name)
                }
            });
        }
        *guard.state = CommandListState::Busy;
        pipeline.command_lists.push(command_list.0.clone());
    }
}

fn map_commands(mut pipeline: ResMut<CommandPipeline>) {
    for command_list in take(&mut pipeline.command_lists) {
        let CommandGuard {
            mut state,
            mut commands,
        } = command_list.lock();
        let CommandListState::Busy = *state else {
            continue;
        };
        *state = CommandListState::Mapping;

        drop(state); // avoid deadlock inside callback

        let count = Arc::new(AtomicUsize::new(commands.len()));
        let callback = {
            let command_list = command_list.clone();
            move |res| {
                // TODO: handle map error
                println!("mapped?: {:?}", res);
                if count.fetch_sub(1, atomic::Ordering::Relaxed) == 0 {
                    *command_list.state.lock() = CommandListState::Done;
                }
            }
        };

        for command in commands.iter_mut() {
            command.async_finish(Box::new(callback.clone()));
        }
    }
}

struct LayoutAndPipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct CommandPipeline {
    map: HashMap<&'static str, LayoutAndPipeline>,
    command_lists: Vec<SharedCommandListData>,
}

impl FromWorld for CommandPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>().wgpu_device();
        let pipeline_cache = world.resource::<PipelineCache>();
        let shader = world.resource::<AssetServer>().load("shaders/vox.wgsl");
        let mut map = HashMap::new();

        let mut create_pipeline = |entry_point: &'static str, layout: wgpu::BindGroupLayout| {
            let layout: BindGroupLayout = layout.into();
            let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some((entry_point.to_owned() + "_pipeline").into()),
                layout: vec![layout.clone()],
                push_constant_ranges: Vec::new(),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from(entry_point),
            });
            map.insert(entry_point, LayoutAndPipeline { layout, pipeline });
        };
        create_pipeline(
            GenerateMeshCommand::ENTRY_POINT,
            GenerateMeshCommand::bind_group_layout(device),
        );
        create_pipeline(
            GeometryCommand::PASTE_SPHERE_ENTRY_POINT,
            GeometryCommand::bind_group_layout(device),
        );
        Self {
            map,
            command_lists: default(),
        }
    }
}

#[derive(Default)]
struct VoxelCommandListsNode;

impl render_graph::Node for VoxelCommandListsNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CommandPipeline>();
        let encoder = render_context.command_encoder();
        for command_list in pipeline.command_lists.iter() {
            let guard = command_list.lock();
            let CommandListState::Busy = *guard.state else {
                continue;
            };
            for command in guard.commands.iter() {
                command.add_pass(encoder, &mut |name| {
                    if let Some(entry) = pipeline.map.get(name) {
                        // TODO: handle pipeline not yet available
                        pipeline_cache.get_compute_pipeline(entry.pipeline).unwrap()
                    } else {
                        panic!("Unknown pipeline in commands: {}", name)
                    }
                });
                command.add_copy(encoder);
            }
        }

        Ok(())
    }
}

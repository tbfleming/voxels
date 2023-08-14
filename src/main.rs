use bevy::{
    pbr::wireframe::Wireframe,
    prelude::*,
    render::{
        render_resource::{PrimitiveTopology, WgpuFeatures},
        settings::WgpuSettings,
        RenderPlugin,
    },
};
use bevy_editor_pls::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_screen_diagnostics::{ScreenDiagnosticsPlugin, ScreenFrameDiagnosticsPlugin};
use std::sync::Arc;

use voxel::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                wgpu_settings: WgpuSettings {
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                },
            }),
            EditorPlugin::default(),
            PanOrbitCameraPlugin,
            ScreenDiagnosticsPlugin::default(),
            ScreenFrameDiagnosticsPlugin,
            VoxelPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut empty_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    empty_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());

    // let _dump = Arc::new(|grid: VoxelGridVec| println!("\n{:#?}\n", &grid.data[..6]));
    let _dump = Arc::new(|grid: VoxelGridVec| println!("\n{:#?}\n", &grid.data));

    let grid_size = UVec3::new(20, 20, 10);
    let grid_center = grid_size / 2;
    let diameter = 10;
    let radius = diameter / 2;
    let circle_center = UVec3::splat(radius);
    let grid = SharedVoxelGrid::new();
    let generate_mesh = GenerateMesh::new();
    let voxel_commands = vec![
        CreateGridCommand::new(grid.clone(), grid_size).boxed(),
        // GeometryCommand::sphere(grid.clone(), diameter, default(), PASTE, 1).boxed(),
        GeometryCommand::cube(grid.clone(), UVec3::new(20, 20, 10), default(), PASTE, 1).boxed(),
        GeometryCommand::sphere(
            grid.clone(),
            diameter,
            (grid_center - circle_center).as_ivec3(),
            PASTE,
            0,
        )
        .boxed(),
        // GetVoxelsCommand::new(grid.clone(), _dump.clone()).boxed(),
        generate_mesh.create_command(grid.clone()).boxed(),
    ];

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(empty_mesh),
            material: materials.add(Color::rgba(0.0, 0.7, 0.7, 1.0).into()),
            transform: Transform::from_xyz(
                -(grid_size.x as f32) / 2.0,
                -(grid_size.y as f32) / 2.0,
                -(grid_size.z as f32) / 2.0,
            ),
            ..default()
        },
        Wireframe,
        generate_mesh,
        VoxelCommandList::new(voxel_commands),
    ));
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_xyz(3.5 + 20.0, 2.5 + 20.0, 5.0 + 20.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: default(),
        brightness: 0.4,
    });
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(3.5 + 20.0, 2.5 + 20.0, 5.0 + 20.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera::default(),
    ));
}

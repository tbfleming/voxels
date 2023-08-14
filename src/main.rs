use bevy::{
    // pbr::wireframe::Wireframe,
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
use rand::prelude::*;

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
        .add_systems(Update, generate_grid)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut empty_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    empty_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());

    let grid_size = UVec3::new(40, 40, 10);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(empty_mesh),
            material: materials.add(Color::rgba(0.8, 0.8, 0.0, 1.0).into()),
            transform: Transform::from_xyz(
                -(grid_size.x as f32) / 2.0,
                -(grid_size.y as f32) / 2.0,
                -(grid_size.z as f32) / 2.0,
            ),
            ..default()
        },
        // Wireframe,
        Stage { size: grid_size },
        GenerateMesh::new(),
        VoxelCommandList::new(Vec::new()),
    ));

    let mut rng = thread_rng();
    for _ in 0..40 {
        commands.spawn((
            Circle {
                diameter: rng.gen_range(1.0..20.0),
                position: Vec3::new(
                    rng.gen_range(0.0..grid_size.x as f32),
                    rng.gen_range(0.0..grid_size.y as f32),
                    rng.gen_range(0.0..grid_size.z as f32),
                ),
                speed: Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ),
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
    }

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

#[derive(Debug, Component)]
struct Stage {
    size: UVec3,
}

#[derive(Debug, Component)]
struct Circle {
    diameter: f32,
    position: Vec3,
    speed: Vec3,
}

fn generate_grid(
    mut stage_query: Query<(&mut Stage, &GenerateMesh, &VoxelCommandList)>,
    mut circles: Query<&mut Circle>,
) {
    let (stage, generate_mesh, voxel_command_list) = stage_query.single_mut();
    let Some(mut voxel_commands) = voxel_command_list.commands_mut() else {
        return;
    };

    let grid = SharedVoxelGrid::new();
    *voxel_commands = vec![
        CreateGridCommand::new(grid.clone(), stage.size).boxed(),
        GeometryCommand::cube(grid.clone(), stage.size, default(), PASTE, 1).boxed(),
    ];

    for mut circle in circles.iter_mut() {
        voxel_commands.push(
            GeometryCommand::sphere(
                grid.clone(),
                circle.diameter as u32,
                stage.size.as_ivec3() / 2 + circle.position.as_ivec3()
                    - IVec3::splat(circle.diameter as i32 / 2),
                PASTE,
                0,
            )
            .boxed(),
        );
        let speed = circle.speed;
        circle.position += speed;
        if circle.position.x < -20.0 || circle.position.x > stage.size.x as f32 {
            circle.speed.x *= -1.0;
        }
        if circle.position.y < -20.0 || circle.position.y > stage.size.y as f32 {
            circle.speed.y *= -1.0;
        }
        if circle.position.z < -20.0 || circle.position.z > stage.size.z as f32 {
            circle.speed.z *= -1.0;
        }
    }

    voxel_commands.push(generate_mesh.create_command(grid.clone()).boxed());
    drop(voxel_commands);
    voxel_command_list.run_again();
}

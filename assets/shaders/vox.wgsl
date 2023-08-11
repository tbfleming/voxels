alias face = array<vec3<f32>, 6>;

@group(0) @binding(0)
var<uniform> voxel_grid_a_size: vec3<u32>;

@group(0) @binding(1)
var<storage,read> voxel_grid_a: array<u32>;

@group(0) @binding(2)
var<uniform> voxel_grid_b_size: vec3<u32>;

@group(0) @binding(3)
var<storage,read> voxel_grid_b: array<u32>;

@group(0) @binding(4)
var<uniform> voxel_grid_out_size: vec3<u32>;

@group(0) @binding(5)
var<storage,read_write> voxel_grid_out: array<u32>;

// Each voxel face is 6 vertices (2 triangles)
@group(0) @binding(6)
var<storage,read_write> mesh: array<vec3<f32>>;

@group(0) @binding(7)
var<storage,read_write> mesh_normals: array<vec3<f32>>;

// Each bit in face_filled represents a face. If the bit is set, the face is filled.
// The 30 LSBs of face_filled[0] represent the first 30 faces.
// The 30 LSBs of face_filled[1] represent the next 30 faces.
//
// Each generate_mesh() invocation converts 5 voxels to 30 faces and fills 1
// entry of face_filled. This separation keeps the shader from having to do
// any atomic operations. I don't know if this approach is actually faster
// than using atomics, but wgpu/naga doesn't support atomics in wgsl yet.
//
// The order of faces within voxels, and the order of voxels within (face_filled
// and mesh) may change in the future; consumers shouldn't rely on it.
@group(0) @binding(8)
var<storage,read_write> face_filled: array<u32>;

struct voxel {
    corner: vec3<f32>,
    material: u32,
}

fn read_voxel(pos: vec3<i32>) -> voxel {
    let index = //
        (pos.x + 1) + //
        (pos.y + 1) * i32(voxel_grid_a_size.x + 2u) + //
        (pos.z + 1) * i32((voxel_grid_a_size.x + 2u) * (voxel_grid_a_size.y + 2u));
    let raw = voxel_grid_a[index];
    let unpacked = unpack4x8snorm(raw);
    return voxel(vec3<f32>(
        unpacked.x * 127.0 / 64.0,
        unpacked.y * 127.0 / 64.0,
        unpacked.z * 127.0 / 64.0
    ), raw >> 24u);
}

fn write_face(pos: vec3<f32>, index: i32, filled: bool, face: face) {
    if filled {
        face_filled[index / 30] |= 1u << (u32(index) % 30u);
        mesh[index * 6 + 0] = pos + face[0];
        mesh[index * 6 + 1] = pos + face[1];
        mesh[index * 6 + 2] = pos + face[2];
        mesh[index * 6 + 3] = pos + face[3];
        mesh[index * 6 + 4] = pos + face[4];
        mesh[index * 6 + 5] = pos + face[5];

        let normal0 = normalize(cross(face[1] - face[0], face[2] - face[0]));
        let normal1 = normalize(cross(face[4] - face[3], face[5] - face[3]));
        mesh_normals[index * 6 + 0] = normal0;
        mesh_normals[index * 6 + 1] = normal0;
        mesh_normals[index * 6 + 2] = normal0;
        mesh_normals[index * 6 + 3] = normal1;
        mesh_normals[index * 6 + 4] = normal1;
        mesh_normals[index * 6 + 5] = normal1;
    }
}

// Each invocation converts 5 voxels (30 faces) and fills 1 entry of face_filled.
// Each workgroup converts a little more than a 6*7*7 cube of voxels.
@compute @workgroup_size(64)
fn generate_mesh(@builtin(global_invocation_id) invocation: vec3<u32>) {
    for (var i = 0u; i < 5u; i += 1u) {
        let voxel_index = invocation.x * 5u + i;
        if voxel_index >= voxel_grid_a_size.x * voxel_grid_a_size.y * voxel_grid_a_size.z {
            break;
        }
        let face_index = i32(voxel_index) * 6;
        let pos_u32 = vec3(
            voxel_index % voxel_grid_a_size.x,
            (voxel_index / voxel_grid_a_size.x) % voxel_grid_a_size.y,
            voxel_index / (voxel_grid_a_size.x * voxel_grid_a_size.y)
        );
        let pos_i32 = vec3<i32>(pos_u32);
        let pos_f32 = vec3<f32>(pos_u32);

        let vox_000 = read_voxel(pos_i32 + vec3<i32>(0, 0, 0));
        if vox_000.material == 0u {
            continue;
        }
        let vox_001 = read_voxel(pos_i32 + vec3<i32>(0, 0, 1));
        let vox_010 = read_voxel(pos_i32 + vec3<i32>(0, 1, 0));
        let vox_011 = read_voxel(pos_i32 + vec3<i32>(0, 1, 1));
        let vox_100 = read_voxel(pos_i32 + vec3<i32>(1, 0, 0));
        let vox_101 = read_voxel(pos_i32 + vec3<i32>(1, 0, 1));
        let vox_110 = read_voxel(pos_i32 + vec3<i32>(1, 1, 0));
        let vox_111 = read_voxel(pos_i32 + vec3<i32>(1, 1, 1));
        let vox_00n = read_voxel(pos_i32 + vec3<i32>(0, 0, -1));
        let vox_0n0 = read_voxel(pos_i32 + vec3<i32>(0, -1, 0));
        let vox_n00 = read_voxel(pos_i32 + vec3<i32>(-1, 0, 0));

        let p000 = vec3<f32>(0.0, 0.0, 0.0) + vox_000.corner;
        let p001 = vec3<f32>(0.0, 0.0, 1.0) + vox_001.corner;
        let p010 = vec3<f32>(0.0, 1.0, 0.0) + vox_010.corner;
        let p011 = vec3<f32>(0.0, 1.0, 1.0) + vox_011.corner;
        let p100 = vec3<f32>(1.0, 0.0, 0.0) + vox_100.corner;
        let p101 = vec3<f32>(1.0, 0.0, 1.0) + vox_101.corner;
        let p110 = vec3<f32>(1.0, 1.0, 0.0) + vox_110.corner;
        let p111 = vec3<f32>(1.0, 1.0, 1.0) + vox_111.corner;

        write_face(pos_f32, face_index + 0, vox_001.material == 0u, face(p001, p101, p111, p111, p011, p001)); // z=1
        write_face(pos_f32, face_index + 1, vox_100.material == 0u, face(p101, p100, p110, p110, p111, p101)); // x=1
        write_face(pos_f32, face_index + 2, vox_00n.material == 0u, face(p100, p000, p010, p010, p110, p100)); // z=0
        write_face(pos_f32, face_index + 3, vox_n00.material == 0u, face(p000, p001, p011, p011, p010, p000)); // x=0
        write_face(pos_f32, face_index + 4, vox_010.material == 0u, face(p011, p111, p110, p110, p010, p011)); // y=1
        write_face(pos_f32, face_index + 5, vox_0n0.material == 0u, face(p000, p100, p101, p101, p001, p000)); // y=0
    }
} // generate_mesh

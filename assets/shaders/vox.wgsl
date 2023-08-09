alias face = array<vec3<f32>, 6>;

// Each voxel face is 6 vertices (2 triangles)
@group(0) @binding(0)
var<storage,read_write> mesh: array<vec3<f32>>;

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
@group(0) @binding(1)
var<storage,read_write> face_filled: array<u32>;

@group(0) @binding(2)
var<uniform> voxel_grid_in_size: vec3<u32>;

@group(0) @binding(3)
var<storage,read> voxel_grid_in: array<u32>;

fn write_face(pos: vec3<f32>, index: i32, filled: bool, face: face) {
    if filled {
        face_filled[index / 30] |= 1u << (u32(index) % 30u);
        mesh[index * 6 + 0] = pos + face[0];
        mesh[index * 6 + 1] = pos + face[1];
        mesh[index * 6 + 2] = pos + face[2];
        mesh[index * 6 + 3] = pos + face[3];
        mesh[index * 6 + 4] = pos + face[4];
        mesh[index * 6 + 5] = pos + face[5];
    }
}

// Each invocation converts 5 voxels (30 faces) and fills 1 entry of face_filled
@compute @workgroup_size(64)
fn generate_mesh(@builtin(global_invocation_id) invocation: vec3<u32>) {
    for (var i = 0u; i < 5u; i += 1u) {
        let voxel_index = invocation.x * 5u + i;
        if voxel_index >= voxel_grid_in_size.x * voxel_grid_in_size.y * voxel_grid_in_size.z {
            break;
        }
        let face_index = i32(voxel_index) * 6;
        let pos = vec3(
            voxel_index % voxel_grid_in_size.x,
            (voxel_index / voxel_grid_in_size.x) % voxel_grid_in_size.y,
            voxel_index / (voxel_grid_in_size.x * voxel_grid_in_size.y)
        );
        let pos_f32 = vec3<f32>(pos);

        let p000 = vec3<f32>(0.0, 0.0, 0.0);
        let p001 = vec3<f32>(0.0, 0.0, 1.0);
        let p010 = vec3<f32>(0.0, 1.0, 0.0);
        let p011 = vec3<f32>(0.0, 1.0, 1.0);
        let p100 = vec3<f32>(1.0, 0.0, 0.0);
        let p101 = vec3<f32>(1.0, 0.0, 1.0);
        let p110 = vec3<f32>(1.0, 1.0, 0.0);
        let p111 = vec3<f32>(1.0, 1.0, 1.0);

        write_face(pos_f32, face_index + 0, true, face(p001, p101, p111, p111, p011, p001)); // z=1
        write_face(pos_f32, face_index + 1, true, face(p101, p100, p110, p110, p111, p101)); // x=1
        write_face(pos_f32, face_index + 2, true, face(p100, p000, p010, p010, p110, p100)); // z=0
        write_face(pos_f32, face_index + 3, true, face(p000, p001, p011, p011, p010, p000)); // x=0
        write_face(pos_f32, face_index + 4, true, face(p011, p111, p110, p110, p010, p011)); // y=1
        write_face(pos_f32, face_index + 5, true, face(p000, p100, p101, p101, p001, p000)); // y=0
    }
}

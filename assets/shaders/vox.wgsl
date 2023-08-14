alias face = array<vec3<f32>, 6>;

const PASTE_MATERIAL        = 1u;  // Copy material of occupied voxels
const PASTE_MATERIAL_ARG    = 2u;  // Set material of occupied voxels to args.material
const PASTE_VERTEXES        = 4u;  // Copy vertexes on the corners of occupied voxels

// Arguments for shaders. See each entry point for details.
struct args_t {
    a_size: vec3<u32>,
    b_size: vec3<u32>,
    out_size: vec3<u32>,
    offset: vec3<i32>,
    size: vec3<u32>,
    flags: u32,
    material: u32,
    diameter: u32,
}

@group(0) @binding(0)
var<uniform> args: args_t;

// See VoxelGridVec for format
@group(0) @binding(1)
var<storage,read> voxel_grid_a: array<u32>;

// See VoxelGridVec for format
@group(0) @binding(2)
var<storage,read> voxel_grid_b: array<u32>;

// See VoxelGridVec for format
@group(0) @binding(3)
var<storage,read_write> voxel_grid_out: array<u32>;

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
// and mesh) may change; consumers shouldn't rely on it.
@group(0) @binding(4)
var<storage,read_write> face_filled: array<u32>;

// Each voxel face is 6 vertices (2 triangles)
@group(0) @binding(5)
var<storage,read_write> mesh: array<vec3<f32>>;

@group(0) @binding(6)
var<storage,read_write> mesh_normals: array<vec3<f32>>;

struct voxel {
    corner: vec3<f32>,
    material: u32,
}

// Calculate the index of a voxel in a voxel grid. Skips begining padding, but -1 can get to it.
fn index(size: vec3<u32>, pos: vec3<i32>) -> i32 {
    return (pos.x + 1) + (pos.y + 1) * i32(size.x + 2u) + (pos.z + 1) * i32((size.x + 2u) * (size.y + 2u));
}

// Pack a voxel into a u32
fn pack(v: voxel) -> u32 {
    let packed = pack4x8snorm(vec4(
        v.corner.x / 127.0 * 64.0,
        v.corner.y / 127.0 * 64.0,
        v.corner.z / 127.0 * 64.0,
        0.0
    ));
    return (packed & 0x00ffffffu) | (v.material << 24u);
}

// Unpack a voxel from a u32
fn unpack(raw: u32) -> voxel {
    let unpacked = unpack4x8snorm(raw);
    return voxel(vec3<f32>(
        unpacked.x * 127.0 / 64.0,
        unpacked.y * 127.0 / 64.0,
        unpacked.z * 127.0 / 64.0
    ), raw >> 24u);
}

fn raw_voxel_a(pos: vec3<i32>) -> u32 {
    return voxel_grid_a[index(args.a_size, pos)];
}

fn filled(raw: u32) -> bool {
    return (raw & 0xff000000u) != 0u;
}

fn unpack_voxel_a(pos: vec3<i32>) -> voxel {
    return unpack(raw_voxel_a(pos));
}

fn write_voxel_out(pos: vec3<i32>, v: voxel) {
    voxel_grid_out[index(args.out_size, pos)] = pack(v);
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

// Generate mesh from voxel_grid_a. Fills face_filled, mesh, and mesh_normals.
// face_filled must be 0-initialized before calling this; mesh and mesh_normals
// don't need to be initialized.
//
// args: {
//      a_size:   size of voxel_grid_a
// }
//
// Each invocation converts 5 voxels (30 faces) and fills 1 entry of face_filled.
// Each workgroup converts a little more than a 6*7*7 cube of voxels.
//
// This needs ceil((args.a_size.x * args.a_size.y * args.a_size.z) / 64) workgroups.
@compute @workgroup_size(64)
fn generate_mesh(@builtin(global_invocation_id) invocation: vec3<u32>) {
    for (var i = 0u; i < 5u; i += 1u) {
        let voxel_index = invocation.x * 5u + i;
        if voxel_index >= args.a_size.x * args.a_size.y * args.a_size.z {
            break;
        }
        let face_index = i32(voxel_index) * 6;
        let pos_u32 = vec3(
            voxel_index % args.a_size.x,
            (voxel_index / args.a_size.x) % args.a_size.y,
            voxel_index / (args.a_size.x * args.a_size.y)
        );
        let pos_i32 = vec3<i32>(pos_u32);
        let pos_f32 = vec3<f32>(pos_u32);

        let vox_000 = unpack_voxel_a(pos_i32 + vec3<i32>(0, 0, 0));
        if vox_000.material == 0u {
            continue;
        }
        let vox_001 = unpack_voxel_a(pos_i32 + vec3<i32>(0, 0, 1));
        let vox_010 = unpack_voxel_a(pos_i32 + vec3<i32>(0, 1, 0));
        let vox_011 = unpack_voxel_a(pos_i32 + vec3<i32>(0, 1, 1));
        let vox_100 = unpack_voxel_a(pos_i32 + vec3<i32>(1, 0, 0));
        let vox_101 = unpack_voxel_a(pos_i32 + vec3<i32>(1, 0, 1));
        let vox_110 = unpack_voxel_a(pos_i32 + vec3<i32>(1, 1, 0));
        let vox_111 = unpack_voxel_a(pos_i32 + vec3<i32>(1, 1, 1));
        let vox_00n = unpack_voxel_a(pos_i32 + vec3<i32>(0, 0, -1));
        let vox_0n0 = unpack_voxel_a(pos_i32 + vec3<i32>(0, -1, 0));
        let vox_n00 = unpack_voxel_a(pos_i32 + vec3<i32>(-1, 0, 0));

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

struct paste_state {
    src_size: vec3<u32>,
    src_pos: vec3<i32>,
    dest_pos: vec3<i32>,
    raw: u32,
}

fn paste_begin(voxel_index: i32, state: ptr<function, paste_state>) -> bool {
    // Include ending padding from source so we get all vertexes
    let scan_size = vec3(i32((*state).src_size.x) + 1, i32((*state).src_size.y) + 1, i32((*state).src_size.z) + 1);
    if voxel_index >= scan_size.x * scan_size.y * scan_size.z {
        return false;
    }
    (*state).src_pos = vec3(
        voxel_index % scan_size.x,
        (voxel_index / scan_size.x) % scan_size.y,
        voxel_index / (scan_size.x * scan_size.y)
    );
    (*state).dest_pos = (*state).src_pos + args.offset;

    // Skip if dest is out of bounds. Allow ending padding.
    if (*state).dest_pos.x < 0 || (*state).dest_pos.y < 0 || (*state).dest_pos.z < 0 || //
        (*state).dest_pos.x > i32(args.out_size.x) || //
        (*state).dest_pos.y > i32(args.out_size.y) || //
        (*state).dest_pos.z > i32(args.out_size.z) {
        return false;
    }

    (*state).raw = voxel_grid_out[index(args.out_size, (*state).dest_pos)];
    return true;
}

// Paste material if dest isn't in padding
fn paste_material(state: ptr<function, paste_state>, src_mat: u32) {
    if (*state).dest_pos.x < i32(args.out_size.x) && //
       (*state).dest_pos.y < i32(args.out_size.y) && //
       (*state).dest_pos.z < i32(args.out_size.z) {
        if (args.flags & PASTE_MATERIAL_ARG) != 0u {
            (*state).raw = ((*state).raw & 0x00ffffffu) | (args.material << 24u);
        } else if (args.flags & PASTE_MATERIAL) != 0u {
            (*state).raw = ((*state).raw & 0x00ffffffu) | (src_mat << 24u);
        }
    }
}

fn paste_vertex(state: ptr<function, paste_state>, src_raw: u32) {
    (*state).raw = ((*state).raw & 0xff000000u) | (src_raw & 0x00ffffffu);
}

fn paste_end(state: ptr<function, paste_state>) {
    voxel_grid_out[index(args.out_size, (*state).dest_pos)] = (*state).raw;
}

// Paste cube into voxel_grid_out. The cube will be centered on
// (args.offset + vec3(diameter/2, diameter/2, diameter/2)).
//
// args: {
//     out_size:    Size of voxel_grid_out
//     offset:      Offset cube's coordinates
//     flags:       Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
//                  Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
//     material:    Material to paste
//     size:        Size of cube
// }
//
// This needs ceil(((args.size.x+1) * (args.size.y+1) * (args.size.z+1)) / 64) workgroups.
@compute @workgroup_size(64)
fn paste_cube(@builtin(global_invocation_id) invocation: vec3<u32>) {
    var state = paste_state(args.size, vec3(0, 0, 0), vec3(0, 0, 0), 0u);
    if !paste_begin(i32(invocation.x), &state) {
        return;
    }
    if state.src_pos.x < i32(args.size.x) && state.src_pos.y < i32(args.size.y) && state.src_pos.z < i32(args.size.z) {
        paste_material(&state, args.material);
    }
    if state.src_pos.x <= i32(args.size.x) && state.src_pos.y <= i32(args.size.y) && state.src_pos.z <= i32(args.size.z) {
        paste_vertex(&state, 0u);
    }
    paste_end(&state);
}

fn sphere_inside(pos: vec3<i32>, size: u32) -> bool {
    let r = f32(size) / 2.0;
    let d = vec3(f32(pos.x) + 0.5 - r, f32(pos.y) + 0.5 - r, f32(pos.z) + 0.5 - r);
    return d.x * d.x + d.y * d.y + d.z * d.z < r * r;
}

fn sphere_include_vertex(pos: vec3<i32>, size: u32) -> bool {
    let count = //
        u32(sphere_inside(pos + vec3(-1, -1, -1), size)) + //
        u32(sphere_inside(pos + vec3(-1, -1, 0), size)) + //
        u32(sphere_inside(pos + vec3(-1, 0, -1), size)) + //
        u32(sphere_inside(pos + vec3(-1, 0, 0), size)) + //
        u32(sphere_inside(pos + vec3(0, -1, -1), size)) + //
        u32(sphere_inside(pos + vec3(0, -1, 0), size)) + //
        u32(sphere_inside(pos + vec3(0, 0, -1), size)) + //
        u32(sphere_inside(pos + vec3(0, 0, 0), size));
    return count != 0u && count != 8u;
}

fn sphere_vertex_center_dist(pos: vec3<i32>, r: f32) -> f32 {
    let d = vec3(f32(pos.x) - r, f32(pos.y) - r, f32(pos.z) - r);
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

fn sphere_vertex_delta(p: i32, r: f32, factor: f32) -> u32 {
    return u32(i32(clamp(round(((f32(p) - r) * factor + r - f32(p)) * 64.0), -127.0, 127.0))) & 0xffu;
}

fn sphere_vertex(pos: vec3<i32>, size: u32) -> u32 {
    let r = f32(size) / 2.0;
    let factor = r / sphere_vertex_center_dist(pos, r);
    return
        sphere_vertex_delta(pos.x, r, factor) | //
        (sphere_vertex_delta(pos.y, r, factor) << 8u) | //
        (sphere_vertex_delta(pos.z, r, factor) << 16u);
}

// Paste sphere into voxel_grid_out. The sphere will be centered on
// (args.offset + vec3(diameter/2, diameter/2, diameter/2)).
//
// args: {
//     out_size:    Size of voxel_grid_out
//     offset:      Offset sphere's coordinates
//     flags:       Any of: PASTE_MATERIAL, PASTE_MATERIAL_ARG, PASTE_VERTEXES.
//                  Note: PASTE_MATERIAL_ARG and PASTE_MATERIAL act the same.
//     material:    Material to paste
//     diameter:    Diameter of sphere
// }
//
// This needs ceil(((args.diameter+1) * (args.diameter+1) * (args.diameter+1)) / 64) workgroups.
@compute @workgroup_size(64)
fn paste_sphere(@builtin(global_invocation_id) invocation: vec3<u32>) {
    var state = paste_state(vec3(args.diameter, args.diameter, args.diameter), vec3(0, 0, 0), vec3(0, 0, 0), 0u);
    if !paste_begin(i32(invocation.x), &state) {
        return;
    }
    if sphere_inside(state.src_pos, args.diameter) {
        paste_material(&state, args.material);
    }
    if sphere_include_vertex(state.src_pos, args.diameter) {
        paste_vertex(&state, sphere_vertex(state.src_pos, args.diameter));
    }
    paste_end(&state);
}

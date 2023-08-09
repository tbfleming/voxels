alias face = array<vec3<f32>, 6>;

@group(0) @binding(0)
var<storage,read_write> mesh: array<vec3<f32>>;

@group(0) @binding(1)
var<storage,read_write> face_filled: array<u32>;

fn write_face(index: i32, filled: bool, face: face) {
    face_filled[index/32] = face_filled[index/32] | (u32(filled) << (u32(index) % 32u));
    mesh[index * 6 + 0] = face[0];
    mesh[index * 6 + 1] = face[1];
    mesh[index * 6 + 2] = face[2];
    mesh[index * 6 + 3] = face[3];
    mesh[index * 6 + 4] = face[4];
    mesh[index * 6 + 5] = face[5];
}

@compute @workgroup_size(1, 1, 1)
fn foo(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let p000 = vec3<f32>(0.0, 0.0, 0.0);
    let p001 = vec3<f32>(0.0, 0.0, 1.0);
    let p010 = vec3<f32>(0.0, 1.0, 0.0);
    let p011 = vec3<f32>(0.0, 1.0, 1.0);
    let p100 = vec3<f32>(1.0, 0.0, 0.0);
    let p101 = vec3<f32>(1.0, 0.0, 1.0);
    let p110 = vec3<f32>(1.0, 1.0, 0.0);
    let p111 = vec3<f32>(1.0, 1.0, 1.0);

    write_face(0, true, face(p001, p101, p111, p111, p011, p001)); // z=1
    write_face(1, true, face(p101, p100, p110, p110, p111, p101)); // x=1
    write_face(2, true, face(p100, p000, p010, p010, p110, p100)); // z=0
    write_face(3, true, face(p000, p001, p011, p011, p010, p000)); // x=0
    write_face(4, true, face(p011, p111, p110, p110, p010, p011)); // y=1
    write_face(5, true, face(p000, p100, p101, p101, p001, p000)); // y=0
}

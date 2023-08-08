alias face = array<vec3<f32>, 6>;

@group(0) @binding(0)
var<storage,read_write> mesh: array<f32>;

fn write_face(index: i32, face: face) {
    mesh[index * 18 + 0] = face[0].x;
    mesh[index * 18 + 1] = face[0].y;
    mesh[index * 18 + 2] = face[0].z;
    mesh[index * 18 + 3] = face[1].x;
    mesh[index * 18 + 4] = face[1].y;
    mesh[index * 18 + 5] = face[1].z;
    mesh[index * 18 + 6] = face[2].x;
    mesh[index * 18 + 7] = face[2].y;
    mesh[index * 18 + 8] = face[2].z;
    mesh[index * 18 + 9] = face[3].x;
    mesh[index * 18 + 10] = face[3].y;
    mesh[index * 18 + 11] = face[3].z;
    mesh[index * 18 + 12] = face[4].x;
    mesh[index * 18 + 13] = face[4].y;
    mesh[index * 18 + 14] = face[4].z;
    mesh[index * 18 + 15] = face[5].x;
    mesh[index * 18 + 16] = face[5].y;
    mesh[index * 18 + 17] = face[5].z;
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

    write_face(0, face(p001, p101, p111, p111, p011, p001)); // z=1
    write_face(1, face(p101, p100, p110, p110, p111, p101)); // x=1
    write_face(2, face(p100, p000, p010, p010, p110, p100)); // z=0
    write_face(3, face(p000, p001, p011, p011, p010, p000)); // x=0
    write_face(4, face(p011, p111, p110, p110, p010, p011)); // y=1
    write_face(5, face(p000, p100, p101, p101, p001, p000)); // y=0
}

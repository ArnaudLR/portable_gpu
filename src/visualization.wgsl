// Full-screen volume ray marcher. The fragment shader reads the D3Q19
// populations produced by the compute pass directly; no GPU readback is used.
struct Parameters {
    dimensions: vec4<u32>,
    omega: f32,
    initial_velocity: f32,
    obstacle_radius: f32,
    _padding: f32,
};

@group(0) @binding(0) var<storage, read> populations: array<f32>;
@group(0) @binding(1) var<uniform> params: Parameters;

const DIRECTIONS: array<vec3<f32>, 19> = array<vec3<f32>, 19>(
    vec3<f32>(0,0,0), vec3<f32>(1,0,0), vec3<f32>(-1,0,0),
    vec3<f32>(0,1,0), vec3<f32>(0,-1,0), vec3<f32>(0,0,1), vec3<f32>(0,0,-1),
    vec3<f32>(1,1,0), vec3<f32>(-1,-1,0), vec3<f32>(1,-1,0), vec3<f32>(-1,1,0),
    vec3<f32>(1,0,1), vec3<f32>(-1,0,-1), vec3<f32>(1,0,-1), vec3<f32>(-1,0,1),
    vec3<f32>(0,1,1), vec3<f32>(0,-1,-1), vec3<f32>(0,1,-1), vec3<f32>(0,-1,1)
);

struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vertex_main(@builtin(vertex_index) vertex: u32) -> VertexOut {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    var out: VertexOut;
    out.position = vec4<f32>(p[vertex], 0, 1);
    out.uv = p[vertex] * 0.5 + 0.5;
    return out;
}

fn velocity_at(position: vec3<f32>) -> vec3<f32> {
    let dims = params.dimensions.xyz;
    let cell = min(vec3<u32>(clamp(position, vec3<f32>(0), vec3<f32>(0.999)) * vec3<f32>(dims)), dims - 1u);
    let site = (cell.z * dims.y + cell.y) * dims.x + cell.x;
    var density = 0.0;
    var momentum = vec3<f32>(0);
    for (var i = 0u; i < 19u; i += 1u) {
        let f = populations[site * 19u + i];
        density += f;
        momentum += f * DIRECTIONS[i];
    }
    return momentum / max(density, 0.0001);
}

@fragment fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Orthographic view through z. Accumulate velocity magnitude and use its
    // direction for a blue/cyan/orange scientific colour map.
    var colour = vec3<f32>(0.008, 0.012, 0.025);
    var alpha = 0.0;
    for (var z = 0u; z < 32u; z += 1u) {
        let v = velocity_at(vec3<f32>(in.uv.x, 1.0 - in.uv.y, (f32(z) + 0.5) / 32.0));
        let speed = length(v);
        let density = smoothstep(0.005, 0.075, speed) * 0.10;
        let mapped = mix(vec3<f32>(0.03, 0.25, 0.85), vec3<f32>(1.0, 0.35, 0.03), clamp(speed / 0.09, 0.0, 1.0));
        colour += (1.0 - alpha) * density * mapped;
        alpha += (1.0 - alpha) * density;
    }
    // Draw the spherical obstacle in neutral grey.
    let center = vec2<f32>(0.25, 0.5);
    if (distance(vec2<f32>(in.uv.x, 1.0 - in.uv.y), center) < params.obstacle_radius / f32(params.dimensions.y)) {
        colour = mix(colour, vec3<f32>(0.65), 0.65);
    }
    return vec4<f32>(colour, 1.0);
}

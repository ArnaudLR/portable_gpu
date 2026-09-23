// D3Q19 BGK lattice-Boltzmann step, using pull streaming and halfway
// bounce-back at the y walls and at a spherical obstacle.
struct Parameters {
    dimensions: vec4<u32>, // x, y, z, reserved
    omega: f32,
    initial_velocity: f32,
    obstacle_radius: f32,
    _padding: f32,
};

@group(0) @binding(0) var<storage, read> populations_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> populations_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Parameters;

const DIRECTIONS: array<vec3<i32>, 19> = array<vec3<i32>, 19>(
    vec3<i32>( 0, 0, 0),
    vec3<i32>( 1, 0, 0), vec3<i32>(-1, 0, 0),
    vec3<i32>( 0, 1, 0), vec3<i32>( 0,-1, 0),
    vec3<i32>( 0, 0, 1), vec3<i32>( 0, 0,-1),
    vec3<i32>( 1, 1, 0), vec3<i32>(-1,-1, 0),
    vec3<i32>( 1,-1, 0), vec3<i32>(-1, 1, 0),
    vec3<i32>( 1, 0, 1), vec3<i32>(-1, 0,-1),
    vec3<i32>( 1, 0,-1), vec3<i32>(-1, 0, 1),
    vec3<i32>( 0, 1, 1), vec3<i32>( 0,-1,-1),
    vec3<i32>( 0, 1,-1), vec3<i32>( 0,-1, 1)
);

const WEIGHTS: array<f32, 19> = array<f32, 19>(
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);

const OPPOSITE: array<u32, 19> = array<u32, 19>(
    0u, 2u, 1u, 4u, 3u, 6u, 5u,
    8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u,
    16u, 15u, 18u, 17u
);

fn site_index(x: u32, y: u32, z: u32, dims: vec3<u32>) -> u32 {
    return (z * dims.y + y) * dims.x + x;
}

fn is_solid(position: vec3<u32>, dims: vec3<u32>) -> bool {
    if (position.y == 0u || position.y + 1u == dims.y) {
        return true;
    }
    let center = vec3<f32>(f32(dims.x) * 0.25, f32(dims.y) * 0.5, f32(dims.z) * 0.5);
    let offset = vec3<f32>(position) - center;
    return dot(offset, offset) <= params.obstacle_radius * params.obstacle_radius;
}

fn equilibrium(direction: u32, density: f32, velocity: vec3<f32>) -> f32 {
    let c_dot_u = dot(vec3<f32>(DIRECTIONS[direction]), velocity);
    let u_squared = dot(velocity, velocity);
    return WEIGHTS[direction] * density *
        (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u_squared);
}

@compute @workgroup_size(4, 4, 4)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = params.dimensions.xyz;
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }

    let site = site_index(id.x, id.y, id.z, dims);
    let position = id;
    if (is_solid(position, dims)) {
        // Solid sites are never sampled by fluid sites; keep them deterministic.
        for (var i = 0u; i < 19u; i += 1u) {
            populations_out[site * 19u + i] = equilibrium(i, 1.0, vec3<f32>(0.0));
        }
        return;
    }

    var streamed: array<f32, 19>;
    var density = 0.0;
    var momentum = vec3<f32>(0.0);

    for (var i = 0u; i < 19u; i += 1u) {
        let direction = DIRECTIONS[i];
        let upstream_y = i32(id.y) - direction.y;
        var incoming: f32;
        if (upstream_y < 0 || upstream_y >= i32(dims.y)) {
            incoming = populations_in[site * 19u + OPPOSITE[i]];
        } else {
            let nx = i32(dims.x);
            let nz = i32(dims.z);
            let upstream = vec3<u32>(
                u32((i32(id.x) - direction.x + nx) % nx),
                u32(upstream_y),
                u32((i32(id.z) - direction.z + nz) % nz)
            );
            if (is_solid(upstream, dims)) {
                incoming = populations_in[site * 19u + OPPOSITE[i]];
            } else {
                let upstream_site = site_index(upstream.x, upstream.y, upstream.z, dims);
                incoming = populations_in[upstream_site * 19u + i];
            }
        }
        streamed[i] = incoming;
        density += incoming;
        momentum += incoming * vec3<f32>(direction);
    }

    let velocity = momentum / density;
    for (var i = 0u; i < 19u; i += 1u) {
        let equilibrium_value = equilibrium(i, density, velocity);
        populations_out[site * 19u + i] = streamed[i] - params.omega * (streamed[i] - equilibrium_value);
    }
}

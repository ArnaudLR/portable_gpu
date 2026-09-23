// Minimal presentation shader. Kept separate from the projection compute shader
// so its texture bind group is group 0 in a compact graphics pipeline layout.
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    var output: VertexOut;
    output.position = vec4<f32>(p[i], 0, 1);
    output.uv = p[i] * 0.5 + 0.5;
    return output;
}

@fragment fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, vec2<f32>(input.uv.x, 1.0 - input.uv.y));
}

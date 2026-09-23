// Minimal presentation shader, intentionally kept simple for older D3D12/Intel
// shader compilers. The projection texture has a fixed 64 x 32 size.
@group(0) @binding(0) var image: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOut {
    // Full-screen triangle without a constant array or dynamic indexing.
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var output: VertexOut;
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

@fragment fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
    let uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(0.999));
    let pixel = vec2<i32>(uv * vec2<f32>(64.0, 32.0));
    return textureLoad(image, pixel, 0);
}

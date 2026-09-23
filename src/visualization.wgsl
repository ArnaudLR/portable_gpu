// GPU-only conversion of D3Q19 populations to a 2D volume projection, followed
// by a deliberately simple texture renderer (friendly to older Intel drivers).
struct Parameters {
    dimensions: vec4<u32>, omega: f32, initial_velocity: f32,
    obstacle_radius: f32, _padding: f32,
};
@group(0) @binding(0) var<storage, read> populations: array<f32>;
@group(0) @binding(1) var<uniform> params: Parameters;
@group(0) @binding(2) var projection: texture_storage_2d<rgba8unorm, write>;

const D: array<vec3<f32>, 19> = array<vec3<f32>, 19>(
 vec3<f32>(0,0,0),vec3<f32>(1,0,0),vec3<f32>(-1,0,0),vec3<f32>(0,1,0),vec3<f32>(0,-1,0),
 vec3<f32>(0,0,1),vec3<f32>(0,0,-1),vec3<f32>(1,1,0),vec3<f32>(-1,-1,0),vec3<f32>(1,-1,0),
 vec3<f32>(-1,1,0),vec3<f32>(1,0,1),vec3<f32>(-1,0,-1),vec3<f32>(1,0,-1),vec3<f32>(-1,0,1),
 vec3<f32>(0,1,1),vec3<f32>(0,-1,-1),vec3<f32>(0,1,-1),vec3<f32>(0,-1,1));

fn velocity(site: u32) -> vec3<f32> {
 var rho=0.0; var momentum=vec3<f32>(0.0);
 for(var i=0u;i<19u;i+=1u){let f=populations[site*19u+i];rho+=f;momentum+=f*D[i];}
 return momentum/max(rho,0.0001);
}

@compute @workgroup_size(8,8)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
 let dims=params.dimensions.xyz;
 if(id.x>=dims.x || id.y>=dims.y){return;}
 var colour=vec3<f32>(0.008,0.012,0.025); var alpha=0.0;
 for(var z=0u;z<dims.z;z+=1u){
  let site=(z*dims.y+id.y)*dims.x+id.x;
  let speed=length(velocity(site));
  let opacity=smoothstep(0.005,0.075,speed)*0.10;
  let mapped=mix(vec3<f32>(0.03,0.25,0.85),vec3<f32>(1.0,0.35,0.03),clamp(speed/0.09,0.0,1.0));
  colour+=(1.0-alpha)*opacity*mapped; alpha+=(1.0-alpha)*opacity;
 }
 let center=vec2<f32>(f32(dims.x)*0.25,f32(dims.y)*0.5);
 if(distance(vec2<f32>(id.xy),center)<params.obstacle_radius){colour=mix(colour,vec3<f32>(0.65),0.65);}
 textureStore(projection,vec2<i32>(id.xy),vec4<f32>(colour,1.0));
}

@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;
struct VertexOut{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>};
@vertex fn vertex_main(@builtin(vertex_index) i:u32)->VertexOut{
 let p=array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3));
 var o:VertexOut;o.position=vec4<f32>(p[i],0,1);o.uv=p[i]*0.5+0.5;return o;
}
@fragment fn fragment_main(i:VertexOut)->@location(0) vec4<f32>{
 return textureSample(image,image_sampler,vec2<f32>(i.uv.x,1.0-i.uv.y));
}

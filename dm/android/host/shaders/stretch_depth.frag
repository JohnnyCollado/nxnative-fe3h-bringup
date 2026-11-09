#version 450
// A depth image stretched over a depth attachment (nvn_vk_draw.c vkd_stretch): where the driver cannot blit the format
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform R { vec4 rect; } r;
layout(location = 0) in vec2 uv;
void main() { gl_FragDepth = texture(src, r.rect.xy + uv * r.rect.zw).r; }

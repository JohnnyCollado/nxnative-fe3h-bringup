#version 450
// The input rectangle stretched over the output, sampled bilinearly (the DLSS stand-in, nvn_vk_draw.c vkd_dlss)
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform R { vec4 rect; } r;   // the input rectangle in normalized coordinates: x, y, width, height
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() { o = texture(src, r.rect.xy + uv * r.rect.zw); }

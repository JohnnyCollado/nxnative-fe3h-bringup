#version 450
// nvn_vk_exec.c lrz_kick: a triangle over the whole target (the draw is scissored to one pixel). Its depth is the draw's first instance: 1 for the LESS draw, 0 for the
// GREATER one, so neither can pass its own compare. Qualcomm's Adreno 830 driver wrote the draw's depth although the fragment shader discards everything (pixel 0,0 of
// every depth image, logs_mem3/000966), which is the whole image of a 1x1 shadow map
void main() { vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2); gl_Position = vec4(p * 2.0 - 1.0, float(gl_InstanceIndex), 1.0); }

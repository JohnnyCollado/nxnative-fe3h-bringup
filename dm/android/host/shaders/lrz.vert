#version 450
// nvn_vk_exec.c lrz_kick: a triangle over the whole target (the draw is scissored to one pixel)
void main() { vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0); }

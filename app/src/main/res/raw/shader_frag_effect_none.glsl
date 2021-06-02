#version 300 es
precision highp float;
in vec2 v_texCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D s_Texture;
uniform vec2 u_TexSize;
uniform float u_dB;
uniform float u_MaxdB;

vec4 mix_audio_color(vec4 tColor) {
    const float led_w = 0.004;
    if (v_texCoord.x > led_w && v_texCoord.x < 1.0-led_w) { return tColor; }
    // create pixel coordinates
    vec2 uv = vec2(0.1, gl_FragCoord.y) / vec2(1.0, u_TexSize.y);
    // quantize coordinates
    const float bands = 1.0; const float segs = 60.0;
    vec2 p = vec2(floor(uv.x * bands) / bands, floor(uv.y * segs) / segs);
    // read frequency data from first row of texture
    float fft  = (u_MaxdB + u_dB) / u_MaxdB;
    // led color
    vec3 color = mix(vec3(0.0, 2.0, 0.0), vec3(2.0, 0.0, 0.0), sqrt(uv.y));
    // mask for bar graph
    float mask = (p.y < fft) ? 1.0 : 0.2;
    // led shape
    vec2 d = fract((uv - p) * vec2(bands, segs)) - 0.5;
    float led = smoothstep(0.5, 0.35, abs(d.x)) * smoothstep(0.5, 0.35, abs(d.y));
    vec3 ledColor = led * color * mask;
    // output final color
    return vec4(ledColor, 1.0);
}

void main() {
    outColor = mix_audio_color(texture(s_Texture, v_texCoord));
}
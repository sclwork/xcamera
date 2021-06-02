#version 300 es
precision highp float;
in vec2 v_texCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D s_Texture;
uniform vec2 u_TexSize;
uniform float u_dB;
uniform float u_MaxdB;
uniform int u_FaceCount;
uniform vec4 u_FaceRect;
uniform float u_Time;
uniform float u_Boundary;
uniform vec4 u_RPoint;
uniform vec2 u_ROffset;

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

vec2 ripple(vec2 tc, float of, float cx, float cy) {
    float ratio = u_TexSize.y / u_TexSize.x;
    vec2 texCoord = tc * vec2(1.0, ratio);
    vec2 touchXY = vec2(cx, cy) * vec2(1.0, ratio);
    float distance = distance(texCoord, touchXY);
    if ((u_Time - u_Boundary) > 0.0
    && (distance <= (u_Time + u_Boundary))
    && (distance >= (u_Time - u_Boundary))) {
        float x = (distance - u_Time);
        float moveDis=of*x*(x-u_Boundary)*(x+u_Boundary);
        vec2 unitDirectionVec = normalize(texCoord - touchXY);
        texCoord = texCoord + (unitDirectionVec * moveDis);
    }
    texCoord = texCoord / vec2(1.0, ratio);
    return texCoord;
}

vec4 ripple_color() {
    float fx = u_FaceRect.x / u_TexSize.x;
    float fy = u_FaceRect.y / u_TexSize.y;
    float fz = u_FaceRect.z / u_TexSize.x;
    float fw = u_FaceRect.w / u_TexSize.y;
    float cx = (fz + fx) / 2.0;
    float cy = (fw + fy) / 2.0;
    vec2 tc = ripple(v_texCoord, 20.0, cx, cy);
    tc=ripple(tc,u_ROffset.x,u_RPoint.x/u_TexSize.x,u_RPoint.y/u_TexSize.y);
    tc=ripple(tc,u_ROffset.y,u_RPoint.z/u_TexSize.x,u_RPoint.w/u_TexSize.y);
    return texture(s_Texture, tc);
}

void main() {
    outColor = mix_audio_color(ripple_color());
}
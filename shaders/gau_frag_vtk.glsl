#version 430 core

in vec3  frag_color;
in float frag_alpha;
in vec3  frag_conic;
in vec2  frag_coordxy;

uniform int render_mod;

out vec4 FragColor;

void main()
{
    // Billboard: full-colour rectangle, no Gaussian falloff
    if (render_mod == -2) {
        FragColor = vec4(frag_color, 1.f);
        return;
    }

    // Gaussian exponent (EWA splatting)
    float power = -0.5f * (frag_conic.x * frag_coordxy.x * frag_coordxy.x
                         + frag_conic.z * frag_coordxy.y * frag_coordxy.y)
                - frag_conic.y * frag_coordxy.x * frag_coordxy.y;

    if (power > 0.f)
        discard;

    float opacity = min(0.99f, frag_alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;

    FragColor = vec4(frag_color, opacity);

    // Special shading modes
    if (render_mod == -3) {
        FragColor.a = (FragColor.a > 0.22f) ? 1.f : 0.f;
    } else if (render_mod == -4) {
        FragColor.a   = (FragColor.a > 0.22f) ? 1.f : 0.f;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}

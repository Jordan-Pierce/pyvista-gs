#version 430 core

// ── SH coefficients (identical to gau_vert.glsl) ─────────────────────────── //
#define SH_C0  0.28209479177387814f
#define SH_C1  0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

// ── Primitive topology ───────────────────────────────────────────────────── //
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// ── SSBOs (same layout as renderer_ogl / gau_vert.glsl) ─────────────────── //
layout(std430, binding = 0) buffer gaussian_data  { float g_data[]; };
layout(std430, binding = 1) buffer gaussian_order { int   gi[];     };

// ── Uniforms (set each frame by renderer_vtk.py UpdateShaderEvent) ────────── //
uniform mat4  view_matrix;
uniform mat4  projection_matrix;
uniform vec3  hfovxy_focal;   // [htanx, htany, focal_pixels]
uniform vec3  cam_pos;
uniform int   sh_dim;
uniform float scale_modifier;
uniform int   render_mod;     // >=0 max SH band; -1 depth; -2 billboard; -3/-4 ball variants

// ── Outputs to fragment shader ───────────────────────────────────────────── //
out vec3  frag_color;
out float frag_alpha;
out vec3  frag_conic;
out vec2  frag_coordxy;

// ── SSBO field offsets (must match GaussianData.flat() layout) ────────────── //
#define POS_IDX     0
#define ROT_IDX     3
#define SCALE_IDX   7
#define OPACITY_IDX 10
#define SH_IDX      11

// ── Helpers ──────────────────────────────────────────────────────────────── //
vec3 get_vec3(int offset)
{
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    float r = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = mat3(
        1.f - 2.f*(y*y + z*z),   2.f*(x*y - r*z),         2.f*(x*z + r*y),
            2.f*(x*y + r*z), 1.f - 2.f*(x*x + z*z),   2.f*(y*z - r*x),
            2.f*(x*z - r*y),       2.f*(y*z + r*x), 1.f - 2.f*(x*x + y*y)
    );
    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    t.x = clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = clamp(t.y / t.z, -limy, limy) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f
    );
    mat3 W   = transpose(mat3(viewmatrix));
    mat3 T   = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

// ── Main ─────────────────────────────────────────────────────────────────── //
void main()
{
    // gl_PrimitiveIDIn = this point's draw-order index (0..N-1).
    // gi[] maps draw order → Gaussian ID, giving correct back-to-front ordering.
    int boxid     = gi[gl_PrimitiveIDIn];
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    int start     = boxid * total_dim;

    vec4 g_pos        = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view   = view_matrix * g_pos;
    vec4 g_pos_clip   = projection_matrix * g_pos_view;
    g_pos_clip.xyz   /= g_pos_clip.w;
    g_pos_clip.w      = 1.f;

    // Frustum cull with a generous margin (same as original)
    if (any(greaterThan(abs(g_pos_clip.xyz), vec3(1.3f))))
        return;

    vec4  g_rot     = get_vec4(start + ROT_IDX);
    vec3  g_scale   = get_vec3(start + SCALE_IDX);
    float g_opacity = g_data[start + OPACITY_IDX];

    mat3 cov3d  = computeCov3D(g_scale * scale_modifier, g_rot);
    vec2 wh     = 2.f * hfovxy_focal.xy * hfovxy_focal.z; // viewport in pixels
    vec3 cov2d  = computeCov2D(g_pos_view,
                               hfovxy_focal.z, hfovxy_focal.z,
                               hfovxy_focal.x, hfovxy_focal.y,
                               cov3d, view_matrix);

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f) return;
    float det_inv = 1.f / det;
    vec3 conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));
    vec2 quadwh_ndc = quadwh_scr / wh * 2.f;

    // ── Colour ─────────────────────────────────────────────────────────── //
    vec3 color;
    if (render_mod == -1) {
        // Depth visualisation
        float depth = -g_pos_view.z;
        depth = (depth < 0.05f) ? 1.f : depth;
        depth = 1.f / depth;
        color = vec3(depth, depth, depth);
    } else {
        int  sh_start = start + SH_IDX;
        vec3 dir      = normalize(g_pos.xyz - cam_pos);
        color         = SH_C0 * get_vec3(sh_start);

        if (sh_dim > 3 && render_mod >= 1) {
            float x = dir.x, y = dir.y, z = dir.z;
            color += -SH_C1 * y * get_vec3(sh_start + 3)
                   +  SH_C1 * z * get_vec3(sh_start + 6)
                   -  SH_C1 * x * get_vec3(sh_start + 9);

            if (sh_dim > 12 && render_mod >= 2) {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                color +=
                    SH_C2_0 * xy          * get_vec3(sh_start + 12) +
                    SH_C2_1 * yz          * get_vec3(sh_start + 15) +
                    SH_C2_2 * (2*zz-xx-yy)* get_vec3(sh_start + 18) +
                    SH_C2_3 * xz          * get_vec3(sh_start + 21) +
                    SH_C2_4 * (xx-yy)     * get_vec3(sh_start + 24);

                if (sh_dim > 27 && render_mod >= 3) {
                    color +=
                        SH_C3_0 * y * (3*xx - yy)      * get_vec3(sh_start + 27) +
                        SH_C3_1 * xy * z                * get_vec3(sh_start + 30) +
                        SH_C3_2 * y * (4*zz - xx - yy) * get_vec3(sh_start + 33) +
                        SH_C3_3 * z * (2*zz - 3*xx - 3*yy) * get_vec3(sh_start + 36) +
                        SH_C3_4 * x * (4*zz - xx - yy) * get_vec3(sh_start + 39) +
                        SH_C3_5 * z * (xx - yy)         * get_vec3(sh_start + 42) +
                        SH_C3_6 * x * (xx - 3*yy)       * get_vec3(sh_start + 45);
                }
            }
        }
        color += 0.5f;
    }

    // ── Emit billboard quad (triangle strip: BL BR TL TR) ─────────────── //
    // Corners in NDC-offset and pixel-space for fragment shader
    vec2 corners_ndc[4] = vec2[4](
        vec2(-1.f, -1.f), vec2( 1.f, -1.f),
        vec2(-1.f,  1.f), vec2( 1.f,  1.f)
    );

    for (int i = 0; i < 4; i++) {
        vec4 pos  = g_pos_clip;
        pos.xy   += corners_ndc[i] * quadwh_ndc;
        gl_Position  = pos;
        frag_color   = color;
        frag_alpha   = g_opacity;
        frag_conic   = conic;
        frag_coordxy = corners_ndc[i] * quadwh_scr;
        EmitVertex();
    }
    EndPrimitive();
}

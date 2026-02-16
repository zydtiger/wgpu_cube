//! WGSL Shaders for 3D Cube with Blinn-Phong Lighting
//!
//! This shader file contains:
//! - Uniform buffer structures for camera and light
//! - Vertex shader: Transforms vertices to clip space
//! - Fragment shader: Calculates Blinn-Phong lighting
//!
//! ## Rendering Pipeline
//! ```text
//! Vertex Shader                    Fragment Shader
//! ┌──────────────┐                 ┌──────────────┐
//! │ position     │                 │ world_pos    │
//! │ normal   ────┼──► MVP ───►     │ normal       │
//! │              │    transform    │              │
//! └──────────────┘                 │ Light Dir ───┼──► Color
//!                                  │ View Dir     │
//!                                  │              │
//!                                  └──────────────┘
//! ```

// ============================================================================
// MARK: Uniform Buffer Structures
// ============================================================================

/// Camera uniform buffer (binding 0)
///
/// Contains the combined view-projection matrix that transforms
/// vertices from world space to clip space.
///
/// ## Alignment
/// - mat4x4<f32> has 16-byte alignment (satisfies WGSL requirements)
struct CameraUniform {
    /// View-projection matrix (4x4 column-major)
    view_proj: mat4x4<f32>,
}

/// Light uniform buffer (binding 1)
///
/// Contains light position and color for Blinn-Phong shading.
///
/// ## Memory Layout (must match Rust's LightUniform)
/// ```text
/// Offset  Field      Type       Size
/// 0       position   vec3<f32>  12
/// 12      _padding   f32        4
/// 16      color      vec3<f32>  12
/// 28      _padding2  f32        4
/// Total: 32 bytes
/// ```
///
/// ## Note on Padding
/// WGSL requires vec3 to be aligned to 16 bytes, so we need padding
/// after each vec3 when followed by another field.
struct LightUniform {
    /// Light position in world space
    position: vec3<f32>,

    /// Padding for 16-byte alignment
    _padding: f32,

    /// Light color (RGB)
    color: vec3<f32>,

    /// Padding for struct size alignment
    _padding2: f32,
}

// ============================================================================
// MARK: Resource Bindings
// ============================================================================

/// Camera uniform buffer binding
/// @group(0) @binding(0) - Matches bind group layout entry 0
@group(0) @binding(0) var<uniform> camera: CameraUniform;

/// Light uniform buffer binding
/// @group(0) @binding(1) - Matches bind group layout entry 1
@group(0) @binding(1) var<uniform> light: LightUniform;

// ============================================================================
// MARK: Vertex Shader Structures
// ============================================================================

/// Vertex input from vertex buffer
///
/// These attributes map to the Vertex struct in cube.rs:
/// - @location(0) = position (Float32x3)
/// - @location(1) = normal (Float32x3)
struct VertexInput {
    /// Vertex position in model space
    @location(0) position: vec3<f32>,

    /// Vertex normal for lighting calculations
    @location(1) normal: vec3<f32>,
}

/// Vertex output to fragment shader
///
/// Data passed from vertex shader to fragment shader.
/// The @builtin(position) is the clip-space position used for rasterization.
/// Other @location outputs are interpolated across the triangle.
struct VertexOutput {
    /// Clip-space position (required for rasterization)
    @builtin(position) clip_position: vec4<f32>,

    /// World-space position (for lighting calculations)
    @location(0) world_position: vec3<f32>,

    /// Interpolated normal (for per-fragment lighting)
    @location(1) world_normal: vec3<f32>,
}

// ============================================================================
// MARK: Vertex Shader
// ============================================================================

/// Vertex shader entry point
///
/// Transforms vertices from model space to clip space and passes
/// world-space data to the fragment shader for lighting.
///
/// ## Transform Pipeline
/// ```text
/// Model Space → (view_proj matrix) → Clip Space
/// ```
///
/// ## Note
/// Since our cube is already centered at origin with no model transform,
/// model space = world space. In a more complex scene, you'd apply
/// a model matrix here.
@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Pass world position (no model transform, so model = world)
    output.world_position = input.position;

    // Pass normal for interpolation
    // Note: For correct lighting with non-uniform scaling, you'd need
    // to transform normals with the inverse-transpose of the model matrix
    output.world_normal = input.normal;

    // Transform position to clip space
    // This applies: view_proj * vec4(position, 1.0)
    // Which is: correction * projection * view * model * position
    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);

    return output;
}

// ============================================================================
// MARK: Fragment Shader
// ============================================================================

/// Fragment shader entry point
///
/// Calculates per-pixel lighting using the Blinn-Phong reflectance model.
///
/// ## Blinn-Phong Lighting Equation
/// ```text
/// final_color = (ambient + diffuse + specular) × object_color
///
/// ambient  = k_a × light_color
/// diffuse  = k_d × max(dot(N, L), 0) × light_color
/// specular = k_s × pow(max(dot(N, H), 0), shininess) × light_color
///
/// Where:
///   N = surface normal (normalized)
///   L = light direction (from surface to light)
///   V = view direction (from surface to camera)
///   H = half vector = normalize(L + V)
/// ```
///
/// ## Parameters
/// |  Component  | Strength |       Notes        |
/// |-------------|----------|--------------------|
/// | Ambient     | 0.1      | Base illumination  |
/// | Diffuse     | 1.0      | Lambert cosine     |
/// | Specular    | 0.5      | Highlights         |
/// | Shininess   | 32       | Specular exponent  |
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // ========================================================================
    // Material Properties
    // ========================================================================

    // Object base color (light blue-gray)
    let object_color = vec3<f32>(0.6, 0.7, 0.8);

    // ========================================================================
    // Lighting Vectors
    // ========================================================================

    // Normalized surface normal
    // Normalization is important even if normals are pre-normalized
    // because interpolation can denormalize them
    let normal = normalize(input.world_normal);

    // Light direction: from fragment to light source
    let light_dir = normalize(light.position - input.world_position);

    // View direction: from fragment to camera
    // Since our camera is at (0, 0, 3) looking at origin,
    // for a fragment at world_position, the view direction is
    // camera_position - world_position = (0, 0, 3) - world_position
    // Simplified to -world_position when camera is at origin in view space
    let view_dir = normalize(-input.world_position);

    // Half vector for Blinn-Phong (between light and view directions)
    // H = normalize(L + V)
    // This is faster than computing reflection for Phong specular
    let half_dir = normalize(light_dir + view_dir);

    // ========================================================================
    // Ambient Component
    // ========================================================================

    // Ambient: constant base illumination
    // Simulates indirect lighting (light bouncing around the scene)
    let ambient_strength = 0.1;
    let ambient = ambient_strength * light.color;

    // ========================================================================
    // Diffuse Component (Lambert)
    // ========================================================================

    // Diffuse: brightness depends on angle between normal and light
    // max(0, ...) ensures we don't get negative lighting
    // dot = cos(angle) → 1 when facing light, 0 at 90°, negative behind
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = diff * light.color;

    // ========================================================================
    // Specular Component (Blinn-Phong)
    // ========================================================================

    // Specular: bright highlights from light reflection
    // dot(N, H) is equivalent to dot(R, V) in Phong but faster
    // Higher shininess = smaller, sharper highlights
    let specular_strength = 0.5;
    let shininess = 32.0;
    let spec = pow(max(dot(normal, half_dir), 0.0), shininess);
    let specular = specular_strength * spec * light.color;

    // ========================================================================
    // Combine and Output
    // ========================================================================

    // Final color: sum of lighting components × object color
    let result = (ambient + diffuse + specular) * object_color;

    // Return as vec4 with alpha = 1.0 (fully opaque)
    return vec4<f32>(result, 1.0);
}

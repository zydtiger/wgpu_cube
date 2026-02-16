//! Light source for 3D scene illumination
//!
//! This module defines the light source used for Blinn-Phong shading.
//! The light is a point light with position and color parameters.
//!
//! ## Blinn-Phong Lighting Model
//!
//! The lighting is calculated in the fragment shader using:
//!
//!   final_color = (ambient + diffuse + specular) × object_color
//!
//!
//! Where:
//! - Ambient: Base illumination independent of light direction
//! - Diffuse: Lambert cosine law (dot product of normal and light direction)
//! - Specular: Blinn-Phong highlight using half-vector

use glam::Vec3;

// ============================================================================
// MARK: Light Struct
// ============================================================================

/// Point light source in the scene
///
/// A point light emits light equally in all directions from its position.
/// This is used for the Blinn-Phong lighting calculations.
///
/// ## Properties
/// - `position`: World-space position of the light
/// - `color`: RGB color intensity (1.0 = full intensity)
///
/// ## Default Configuration
/// The default light is positioned at (2, 2, 3) - above, to the right,
/// and in front of the cube - with white color for neutral illumination.
pub struct Light {
    /// World-space position of the light source
    pub position: Vec3,

    /// RGB color of the light (values typically 0.0 - 1.0)
    pub color: Vec3,
}

impl Light {
    /// Creates a new light with default settings
    ///
    /// ## Default Values
    /// | Property |         Value         |
    /// |----------|-----------------------|
    /// | position | (2.0, 2.0, 3.0)       |
    /// | color    | (1.0, 1.0, 1.0) white |
    pub fn new() -> Self {
        Self {
            position: Vec3::new(2.0, 2.0, 3.0),
            color: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl Default for Light {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MARK: Light Uniform (GPU Buffer)
// ============================================================================

/// GPU-aligned uniform buffer for light data
///
/// This struct is designed to be uploaded directly to the GPU as a
/// uniform buffer. It contains the light position and color with
/// proper padding for WGSL alignment requirements.
///
/// ## Memory Layout
/// ```text
/// Offset  Size  Field     Alignment
/// 0       12    position  16 (vec3 requires 16-byte alignment start)
/// 12      4     _padding  -
/// 16      12    color     16
/// 28      4     _padding2 -
/// Total:  32 bytes
/// ```
///
/// ## WGSL Alignment Rules
/// In WGSL, `vec3<f32>` has:
/// - Size: 12 bytes (3 × 4)
/// - Alignment: 16 bytes
///
/// This means each vec3 must start at a 16-byte boundary, requiring
/// padding after each vec3 if followed by another value.
///
/// ## Safety
/// - `#[repr(C)]` ensures C-compatible memory layout
/// - `bytemuck::Pod` allows safe casting to byte slices
/// - Padding fields ensure 16-byte alignment
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    /// Light position in world space
    pub position: [f32; 3],

    /// Padding to align `color` to 16-byte boundary
    pub _padding: f32,

    /// Light color (RGB, values 0.0 - 1.0)
    pub color: [f32; 3],

    /// Padding to make struct size a multiple of 16
    pub _padding2: f32,
}

impl LightUniform {
    /// Creates a new LightUniform with default values
    ///
    /// Initializes with the same defaults as `Light::new()`.
    pub fn new() -> Self {
        Self {
            position: [2.0, 2.0, 3.0],
            _padding: 0.0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0.0,
        }
    }

    /// Updates the uniform from a Light struct
    ///
    /// Call this when the light properties change before uploading
    /// to the GPU.
    ///
    /// ## Arguments
    /// - `light`: Reference to the Light to copy values from
    pub fn update(&mut self, light: &Light) {
        self.position = light.position.into();
        self.color = light.color.into();
    }
}

impl Default for LightUniform {
    fn default() -> Self {
        Self::new()
    }
}

//! Cube geometry for 3D rendering
//!
//! This module defines the vertex data and indices for rendering a unit cube
//! (2×2×2 units, centered at origin). Each face has unique vertices with
//! proper normals for flat shading.
//!
//! ## Cube Layout
//! ```text
//!        ┌────────┐
//!       /│       /│
//!      / │  +Y  / │
//!     /  │     /  │
//!    ┌────────┐   │      +Y (top)
//!    │   │    │   │       │
//!    │   └────│───┘       │
//!    │  /     │  /        │
//!    │ /  -Z  │ /  +Z     └───+X (right)
//!    │/       │/         /
//!    └────────┘         /
//!        +X            +Z (front)
//! ```
//!
//! ## Vertex Count
//! - 24 vertices (6 faces × 4 vertices per face)
//! - Each face needs separate vertices because normals must be unique per face
//!
//! ## Index Count
//! - 36 indices (6 faces × 2 triangles × 3 vertices)

use wgpu::VertexAttribute;

// ============================================================================
// MARK: Vertex Struct
// ============================================================================

/// Vertex data for cube rendering
///
/// Each vertex contains a position and a normal vector. The normal is used
/// for lighting calculations in the fragment shader.
///
/// ## Memory Layout
/// ```text
/// Offset  Size  Field
/// 0       12    position (3 × f32)
/// 12      12    normal   (3 × f32)
/// Total:  24 bytes
/// ```
///
/// ## Attributes
/// | Location | Attribute | Format      |
/// |----------|-----------|-------------|
/// | 0        | position  | Float32x3   |
/// | 1        | normal    | Float32x3   |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Position in model space (cube centered at origin, size 2×2×2)
    pub position: [f32; 3],

    /// Normal vector for lighting calculations
    /// Points outward from the cube face
    pub normal: [f32; 3],
}

impl Vertex {
    /// Creates the vertex buffer layout descriptor for wgpu
    ///
    /// This descriptor tells wgpu how to interpret the vertex data
    /// when passing it to the vertex shader.
    ///
    /// ## Returns
    /// A `VertexBufferLayout` describing:
    /// - Array stride: 24 bytes per vertex
    /// - Step mode: Per-vertex (not instanced)
    /// - Two attributes: position (location 0) and normal (location 1)
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            // Size of one vertex in bytes
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,

            // Process one vertex per shader invocation
            step_mode: wgpu::VertexStepMode::Vertex,

            attributes: &[
                // Position attribute @ location(0)
                VertexAttribute {
                    offset: 0, // Starts at beginning of vertex
                    shader_location: 0, // @location(0) in WGSL
                    format: wgpu::VertexFormat::Float32x3, // vec3<f32>
                },
                // Normal attribute @ location(1)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress, // After position
                    shader_location: 1, // @location(1) in WGSL
                    format: wgpu::VertexFormat::Float32x3, // vec3<f32>
                },
            ],
        }
    }
}

// ============================================================================
// MARK: Vertex Data
// ============================================================================

/// Cube vertices: 24 vertices (6 faces × 4 vertices)
///
/// Each face has 4 vertices with the same normal direction.
/// This is necessary for flat shading - if we shared vertices between
/// faces, the normals would be averaged (smooth shading).
///
/// ## Face Layout
/// | Face   | Vertices | Normal   |
/// |--------|----------|----------|
/// | Front  | 0-3      | +Z       |
/// | Back   | 4-7      | -Z       |
/// | Top    | 8-11     | +Y       |
/// | Bottom | 12-15    | -Y       |
/// | Right  | 16-19    | +X       |
/// | Left   | 20-23    | -X       |
///
/// ## Winding Order
/// Vertices are in counter-clockwise (CCW) order when viewed from outside.
/// This matches the `front_face: FrontFace::Ccw` setting in the pipeline.
pub const VERTICES: &[Vertex] = &[
    // =========================================================================
    // Front face (normal: +Z, towards viewer)
    // =========================================================================
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
    },

    // =========================================================================
    // Back face (normal: -Z, away from viewer)
    // =========================================================================
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [0.0, 0.0, -1.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [0.0, 0.0, -1.0],
    },

    // =========================================================================
    // Top face (normal: +Y)
    // =========================================================================
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [0.0, 1.0, 0.0],
    },

    // =========================================================================
    // Bottom face (normal: -Y)
    // =========================================================================
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [0.0, -1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0.0, -1.0, 0.0],
    },

    // =========================================================================
    // Right face (normal: +X)
    // =========================================================================
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [1.0, 0.0, 0.0],
    },

    // =========================================================================
    // Left face (normal: -X)
    // =========================================================================
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [-1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [-1.0, 0.0, 0.0],
    },
];

// ============================================================================
// MARK: Index Data
// ============================================================================

/// Cube indices: 36 indices (6 faces × 2 triangles × 3 vertices)
///
/// Each face is made of 2 triangles (6 indices). The indices reference
/// vertices in the VERTICES array.
///
/// ## Triangle Winding
/// All triangles use counter-clockwise (CCW) winding order for front faces.
///
/// ## Index Pattern
/// Each face uses the pattern: `v0, v1, v2, v2, v3, v0`
/// ```text
/// v3 ─────── v2
///  │  ╲      │
///  │    ╲    │
///  │      ╲  │
/// v0 ─────── v1
/// ```
/// This creates two triangles: (v0, v1, v2) and (v2, v3, v0)
pub const INDICES: &[u16] = &[
    // Front face (vertices 0-3)
    0, 1, 2, 2, 3, 0,

    // Back face (vertices 4-7)
    4, 5, 6, 6, 7, 4,

    // Top face (vertices 8-11)
    8, 9, 10, 10, 11, 8,

    // Bottom face (vertices 12-15)
    12, 13, 14, 14, 15, 12,

    // Right face (vertices 16-19)
    16, 17, 18, 18, 19, 16,

    // Left face (vertices 20-23)
    20, 21, 22, 22, 23, 20,
];

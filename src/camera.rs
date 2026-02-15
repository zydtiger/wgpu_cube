//! Camera system for 3D scene rendering
//!
//! This module provides camera functionality including:
//! - View matrix (world to camera transformation)
//! - Projection matrix (camera to clip space transformation)
//! - GPU-compatible uniform buffer for shaders
//!
//! ## Coordinate Systems
//!
//! ```text
//! World Space → View Space → Clip Space
//!     (view)        (proj)     (correction)
//! ```
//!
//! ### wgpu Coordinate System Notes
//! - wgpu uses a right-handed coordinate system
//! - NDC Z-range is [0, 1] (not [-1, 1] like OpenGL)
//! - A correction matrix is applied to convert from OpenGL conventions

use glam::{Mat4, Vec3};

// ============================================================================
// MARK: Camera Struct
// ============================================================================

/// Camera for viewing the 3D scene
///
/// This struct contains all the parameters needed to construct the
/// view-projection matrix that transforms vertices from world space
/// to clip space.
///
/// ## Parameters
/// - `eye`: Camera position in world space
/// - `target`: Point the camera is looking at
/// - `up`: Up direction (typically Y-axis)
/// - `aspect`: Aspect ratio (width / height)
/// - `fovy`: Vertical field of view in radians
/// - `znear`: Near clipping plane distance
/// - `zfar`: Far clipping plane distance
///
/// ## Example
/// ```ignore
/// let camera = Camera::new(800, 600);
/// let view_proj = camera.build_view_projection_matrix();
/// ```
pub struct Camera {
    /// Camera position in world space
    pub eye: Vec3,

    /// Point the camera is looking at (world space)
    pub target: Vec3,

    /// Up direction vector (defines the camera's orientation)
    pub up: Vec3,

    /// Aspect ratio of the viewport (width / height)
    pub aspect: f32,

    /// Vertical field of view in radians
    pub fovy: f32,

    /// Near clipping plane distance (must be > 0)
    pub znear: f32,

    /// Far clipping plane distance (must be > znear)
    pub zfar: f32,
}

impl Camera {
    /// Creates a new camera with default settings
    ///
    /// The camera is positioned at (0, 0, 3), looking at the origin,
    /// with a 45-degree field of view.
    ///
    /// ## Arguments
    /// - `width`: Viewport width in pixels
    /// - `height`: Viewport height in pixels
    ///
    /// ## Default Values
    /// | Parameter | Value |
    /// |-----------|-------|
    /// | eye | (0, 0, 3) |
    /// | target | (0, 0, 0) |
    /// | up | (0, 1, 0) |
    /// | fovy | 45° (in radians) |
    /// | znear | 0.1 |
    /// | zfar | 100.0 |
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            aspect: width as f32 / height as f32,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        }
    }

    /// Builds the combined view-projection matrix
    ///
    /// This matrix transforms vertices from world space directly to
    /// clip space (normalized device coordinates after perspective divide).
    ///
    /// ## Pipeline
    /// ```text
    /// View Matrix (look_at_rh)
    ///     ↓
    /// Projection Matrix (perspective_rh)
    ///     ↓
    /// Correction Matrix (OpenGL → wgpu Z-range)
    ///     ↓
    /// Final View-Projection Matrix
    /// ```
    ///
    /// ## Coordinate Conversion
    /// The correction matrix converts from OpenGL's Z-range [-1, 1]
    /// to wgpu's Z-range [0, 1]:
    /// ```text
    /// z_wgpu = 0.5 * z_opengl + 0.5
    /// ```
    ///
    /// ## Returns
    /// A 4x4 view-projection matrix ready for upload to GPU
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        // Create view matrix: transforms world space to view (camera) space
        // rh = right-handed coordinate system
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);

        // Create projection matrix: transforms view space to clip space
        // rh = right-handed, depth maps to [znear, zfar] then to NDC
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);

        // Correction matrix: converts OpenGL NDC [-1,1] Z-range to wgpu [0,1]
        //
        // OpenGL: z_ndc = (2 * z_view - near - far) / (far - near)  →  [-1, 1]
        // wgpu:   z_ndc = (z_view - near) / (far - near)            →  [0, 1]
        //
        // This matrix applies: z' = 0.5 * z + 0.5
        let correction = Mat4::from_cols(
            glam::vec4(1.0, 0.0, 0.0, 0.0),
            glam::vec4(0.0, 1.0, 0.0, 0.0),
            glam::vec4(0.0, 0.0, 0.5, 0.0),
            glam::vec4(0.0, 0.0, 0.5, 1.0),
        );

        // Apply transformations: correction * projection * view
        // Order matters! Matrices are applied right-to-left
        correction * proj * view
    }
}

// ============================================================================
// MARK: Camera Uniform (GPU Buffer)
// ============================================================================

/// GPU-aligned uniform buffer for camera data
///
/// This struct is designed to be uploaded directly to the GPU as a
/// uniform buffer. It contains only the view-projection matrix.
///
/// ## Memory Layout
/// - Size: 64 bytes (16 × f32)
/// - Alignment: 16-byte aligned (satisfies WGSL requirements)
///
/// ## Safety
/// - `#[repr(C)]` ensures C-compatible memory layout
/// - `bytemuck::Pod` allows safe casting to byte slices
/// - `bytemuck::Zeroable` allows zero-initialization
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix as a 4x4 array of f32
    /// Layout: column-major (WGSL expects column-major)
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Creates a new CameraUniform initialized to identity matrix
    ///
    /// The identity matrix means no transformation is applied.
    /// Call `update_view_proj()` to set the actual camera transformation.
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    /// Updates the view-projection matrix from a Camera
    ///
    /// Call this whenever the camera changes (resize, movement, etc.)
    /// before uploading to the GPU.
    ///
    /// ## Arguments
    /// - `camera`: Reference to the Camera to extract the matrix from
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().to_cols_array_2d();
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

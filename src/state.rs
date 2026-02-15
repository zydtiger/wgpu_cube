//! wgpu rendering state management
//!
//! This module encapsulates all the wgpu state needed for rendering,
//! including device, queue, surface, pipeline, and buffers.
//!
//! ## Architecture
//!
//! ```text
//! State
//! ├── Surface (swapchain for presenting to window)
//! ├── Device (GPU context for creating resources)
//! ├── Queue (command submission)
//! ├── RenderPipeline (shader + state configuration)
//! ├── Buffers
//! │   ├── Vertex Buffer (cube geometry)
//! │   ├── Index Buffer (triangle indices)
//! │   ├── Camera Buffer (uniform)
//! │   └── Light Buffer (uniform)
//! ├── Bind Groups (resource binding for shaders)
//! └── Depth Texture (depth testing)
//! ```

use std::future::Future;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraUniform};
use crate::cube::{self, Vertex};
use crate::light::{Light, LightUniform};

// ============================================================================
// MARK: State Struct
// ============================================================================

/// Main rendering state container
///
/// This struct holds all the resources needed for rendering a frame.
/// It is created once during application initialization and persists
/// for the lifetime of the application.
///
/// ## Resource Categories
///
/// | Category | Fields | Purpose |
/// |----------|--------|---------|
/// | Core | surface, device, queue, config, size | GPU connection |
/// | Pipeline | render_pipeline | Shader + render state |
/// | Geometry | vertex_buffer, index_buffer, num_indices | Mesh data |
/// | Camera | camera, camera_uniform, camera_buffer | View transformation |
/// | Lighting | light, light_uniform, light_buffer | Illumination |
/// | Binding | bind_group, bind_group_layout | Shader resources |
/// | Depth | depth_texture, depth_texture_view | Depth testing |
pub struct State {
    // ========================================================================
    // Core wgpu Resources
    // ========================================================================
    /// The surface we render to (connected to the window)
    pub surface: wgpu::Surface<'static>,

    /// The GPU device (context for creating resources)
    pub device: wgpu::Device,

    /// The command queue for submitting render commands
    pub queue: wgpu::Queue,

    /// Surface configuration (format, size, vsync, etc.)
    pub config: wgpu::SurfaceConfiguration,

    /// Current window size in physical pixels
    pub size: winit::dpi::PhysicalSize<u32>,

    // ========================================================================
    // Pipeline
    // ========================================================================
    /// The render pipeline (shaders + state configuration)
    pub render_pipeline: wgpu::RenderPipeline,

    // ========================================================================
    // Geometry Buffers
    // ========================================================================
    /// Vertex buffer containing cube vertices
    pub vertex_buffer: wgpu::Buffer,

    /// Index buffer containing triangle indices
    pub index_buffer: wgpu::Buffer,

    /// Number of indices to draw (36 for a cube)
    pub num_indices: u32,

    // ========================================================================
    // Camera Resources
    // ========================================================================
    /// Camera state (position, projection parameters)
    pub camera: Camera,

    /// Camera uniform data ready for GPU upload
    pub camera_uniform: CameraUniform,

    /// GPU buffer containing camera uniform data
    pub camera_buffer: wgpu::Buffer,

    // ========================================================================
    // Light Resources
    // ========================================================================
    /// Light source state
    pub light: Light,

    /// Light uniform data ready for GPU upload
    pub light_uniform: LightUniform,

    /// GPU buffer containing light uniform data
    pub light_buffer: wgpu::Buffer,

    // ========================================================================
    // Binding Resources
    // ========================================================================
    /// Bind group (binds buffers to shader slots)
    pub bind_group: wgpu::BindGroup,

    /// Bind group layout (describes binding structure)
    pub bind_group_layout: wgpu::BindGroupLayout,

    // ========================================================================
    // Depth Resources
    // ========================================================================
    /// Depth texture for depth testing
    pub depth_texture: wgpu::Texture,

    /// View into the depth texture
    pub depth_texture_view: wgpu::TextureView,
}

impl State {
    /// Creates a new State, initializing all wgpu resources
    ///
    /// This is an async function because adapter and device creation
    /// may take time (querying GPU capabilities).
    ///
    /// ## Initialization Steps
    /// 1. Create instance and surface
    /// 2. Request adapter (selects GPU)
    /// 3. Request device and queue
    /// 4. Configure surface
    /// 5. Create shader module
    /// 6. Create uniform buffers
    /// 7. Create bind groups
    /// 8. Create render pipeline
    /// 9. Create geometry buffers
    /// 10. Create depth texture
    ///
    /// ## Arguments
    /// - `window`: The window to render to (wrapped in Arc for shared ownership)
    ///
    /// ## Panics
    /// Panics if any wgpu resource fails to create
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // ====================================================================
        // Create Instance and Surface
        // ====================================================================

        // Instance: The entry point to wgpu, connects to the GPU backend
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Use all available backends
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&descriptor);

        // Surface: The target for rendering (connected to window)
        // Box<dyn WindowHandle> allows different window types
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Window(Box::new(window)))
            .expect("Failed to create surface");

        // ====================================================================
        // Request Adapter
        // ====================================================================

        // Adapter: Represents a physical GPU with specific capabilities
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), // Let system decide
                compatible_surface: Some(&surface),                 // Must work with our surface
                force_fallback_adapter: false,                      // Don't use software renderer
            })
            .await
            .expect("Failed to find adapter");

        // ====================================================================
        // Request Device and Queue
        // ====================================================================

        // Device: Logical GPU context, used to create resources
        // Queue: Command submission interface
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create device");

        // ====================================================================
        // Configure Surface
        // ====================================================================

        // Get surface capabilities (formats, present modes, etc.)
        let surface_caps = surface.get_capabilities(&adapter);

        // Prefer sRGB format for correct color display
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // We'll render to this
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0], // Usually Fifo (vsync)
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2, // Double buffering
        };

        surface.configure(&device, &config);

        // ====================================================================
        // Load Shader
        // ====================================================================

        // Compile the WGSL shader at runtime
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/cube.wgsl").into()),
        });

        // ====================================================================
        // Create Camera Resources
        // ====================================================================

        let camera = Camera::new(config.width, config.height);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        // Create GPU buffer for camera uniforms
        // UNIFORM: Can be bound as uniform buffer
        // COPY_DST: Can be written to via queue.write_buffer
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ====================================================================
        // Create Light Resources
        // ====================================================================

        let light = Light::new();
        let mut light_uniform = LightUniform::new();
        light_uniform.update(&light);

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ====================================================================
        // Create Bind Group Layout
        // ====================================================================

        // Describes the structure of resources bound to the shader
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Binding 0: Camera uniform buffer
                // Visible to both vertex and fragment shaders
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Light uniform buffer
                // Only visible to fragment shader (where lighting is calculated)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ====================================================================
        // Create Bind Group
        // ====================================================================

        // Instantiates the bind group with actual buffers
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
            ],
        });

        // ====================================================================
        // Create Pipeline Layout
        // ====================================================================

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // ====================================================================
        // Create Render Pipeline
        // ====================================================================

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),

            // Vertex stage: Transform vertices to clip space
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[Vertex::desc()], // Vertex format
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },

            // Fragment stage: Calculate pixel colors
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,                  // Match surface format
                    blend: Some(wgpu::BlendState::REPLACE), // No blending
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),

            // Primitive assembly: How vertices form triangles
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // Counter-clockwise = front
                cull_mode: Some(wgpu::Face::Back), // Don't draw back faces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            // Depth/stencil: Enable depth testing
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Closer = in front
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),

            // Multisampling: Disabled (1 sample per pixel)
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },

            multiview_mask: None,
            cache: None,
        });

        // ====================================================================
        // Create Geometry Buffers
        // ====================================================================

        // Vertex buffer: Contains all vertex data (positions, normals)
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(cube::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Index buffer: Defines triangles using vertex indices
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(cube::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = cube::INDICES.len() as u32;

        // ====================================================================
        // Create Depth Texture
        // ====================================================================

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ====================================================================
        // Return State
        // ====================================================================

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera,
            camera_uniform,
            camera_buffer,
            light,
            light_uniform,
            light_buffer,
            bind_group,
            bind_group_layout,
            depth_texture,
            depth_texture_view,
        }
    }

    // ========================================================================
    // MARK: Resize Handler
    // ========================================================================

    /// Handles window resize events
    ///
    /// This method updates the surface configuration, camera aspect ratio,
    /// and recreates the depth texture to match the new window size.
    ///
    /// ## Arguments
    /// - `new_size`: New window dimensions in physical pixels
    ///
    /// ## Note
    /// This method ignores zero-sized dimensions to avoid wgpu errors.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Update stored size and surface config
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update camera aspect ratio
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            self.camera_uniform.update_view_proj(&self.camera);

            // Upload updated camera matrix to GPU
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );

            // Recreate depth texture at new size
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_texture_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    // ========================================================================
    // MARK: Render Function
    // ========================================================================

    /// Renders a single frame
    ///
    /// This method:
    /// 1. Acquires the next swapchain texture
    /// 2. Creates a command encoder
    /// 3. Begins a render pass (clears screen and depth buffer)
    /// 4. Sets pipeline, bind group, and buffers
    /// 5. Draws indexed geometry
    /// 6. Submits commands and presents
    ///
    /// ## Returns
    /// - `Ok(())`: Frame rendered successfully
    /// - `Err(SurfaceError)`: Various surface errors (lost, outdated, OOM, etc.)
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Get the next frame's texture from the swapchain
        let output = self.surface.get_current_texture()?;

        // Create a view into the texture for rendering
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create a command encoder to record render commands
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Begin the render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),

                // Color attachment: The screen
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None, // No multisample resolve
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Clear to dark blue-gray background
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.12,
                            b: 0.18,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],

                multiview_mask: None,

                // Depth attachment: For depth testing
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear to far plane
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),

                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set the render pipeline (shaders + state)
            render_pass.set_pipeline(&self.render_pipeline);

            // Set the bind group (camera + light uniforms)
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            // Set vertex buffer (contains positions and normals)
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            // Set index buffer (defines triangles)
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Draw the cube
            // Arguments: index range, base vertex, instance range
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // Submit commands to the GPU
        self.queue.submit(std::iter::once(encoder.finish()));

        // Present the rendered frame to the screen
        output.present();

        Ok(())
    }
}

// ============================================================================
// MARK: Async Helper
// ============================================================================

/// Blocks on an async future using pollster
///
/// wgpu initialization is async, but our main function is synchronous.
/// This helper bridges that gap by running the future to completion.
///
/// ## Arguments
/// - `future`: The async operation to run
///
/// ## Returns
/// The result of the future
pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    pollster::block_on(future)
}

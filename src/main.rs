//! Main entry point for the wgpu 3D Cube application
//!
//! This module implements the application lifecycle using winit's `ApplicationHandler` trait,
//! which is the recommended approach for winit 0.30+.
//!
//! ## Architecture Overview
//!
//! ```text
//! main()
//!   └── EventLoop (winit's event processing system)
//!         └── App (ApplicationHandler implementation)
//!               ├── Window (display surface)
//!               └── State (wgpu rendering state)
//!                     ├── Device, Queue, Surface
//!                     ├── RenderPipeline
//!                     ├── Buffers (vertex, index, uniforms)
//!                     └── Camera & Light
//! ```

// ============================================================================
// MARK: Module Declarations
// ============================================================================

mod camera; // Camera system with view-projection matrix calculation
mod cube; // Cube geometry (24 vertices, 36 indices)
mod light; // Light source with Blinn-Phong lighting parameters
mod state; // wgpu state management (device, pipeline, buffers)

// ============================================================================
// MARK: Imports
// ============================================================================

use std::sync::Arc;

use state::State;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// ============================================================================
// MARK: Application State
// ============================================================================

/// Main application struct that holds all state
///
/// This struct implements `ApplicationHandler` to receive and process
/// window events from winit's event loop.
///
/// ## Lifecycle
/// 1. `new()` - Creates an empty App with no window or state
/// 2. `resumed()` - Called when the app becomes active; initializes window and GPU state
/// 3. `window_event()` - Called for each window event (resize, keyboard, redraw, etc.)
/// 4. `about_to_wait()` - Called when there are no more events; triggers redraw
struct App {
    /// The display window wrapped in Arc for shared ownership
    /// (needed because wgpu's surface requires a reference to the window)
    window: Option<Arc<Window>>,

    /// The wgpu rendering state (device, queue, pipeline, buffers, etc.)
    /// This is None until the app is resumed, then stays Some for the lifetime
    state: Option<State>,
}

impl App {
    /// Creates a new App instance with uninitialized state
    ///
    /// The window and state are lazily initialized in `resumed()` rather than
    /// here, because window creation requires an `ActiveEventLoop` reference.
    fn new() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

// ============================================================================
// MARK: ApplicationHandler Implementation
// ============================================================================

impl ApplicationHandler for App {
    /// Called when the application is resumed/activated
    ///
    /// This is the initialization point where we create the window and
    /// initialize the wgpu rendering state. On macOS, this may be called
    /// multiple times (e.g., when the app is reactivated after being hidden).
    ///
    /// The `if self.window.is_none()` guard ensures we only initialize once.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create the window with default attributes
            // Arc is used because wgpu's Surface needs to hold a reference to the window
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("wgpu 3D Cube with Blinn-Phong Lighting"),
                    )
                    .expect("Failed to create window"),
            );

            // Initialize the wgpu state asynchronously
            // This creates the GPU device, pipeline, buffers, etc.
            // block_on() runs the async initialization to completion
            let state = pollster::block_on(State::new(window.clone()));
            self.state = Some(state);

            // Store the window reference
            self.window = Some(window);
        }
    }

    /// Called for each window event
    ///
    /// This is the main event handling function. Events include:
    /// - CloseRequested: User clicked the close button
    /// - Resized: Window was resized
    /// - KeyboardInput: Keyboard key pressed/released
    /// - RedrawRequested: Window needs to be redrawn
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Guard: Exit early if state isn't initialized yet
        let Some(state) = &mut self.state else { return };
        let Some(window) = &self.window else { return };

        match event {
            // User requested to close the window (clicked X button, Cmd+W, etc.)
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // Window was resized; update the swapchain and camera aspect ratio
            WindowEvent::Resized(new_size) => {
                state.resize(new_size);
            }

            // Keyboard input event
            // KeyEvent contains: physical_key (hardware-independent), logical_key (layout-dependent), state, etc.
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state: key_state,
                        .. // Ignore other KeyEvent fields (logical_key, text, etc.)
                    },
                .. // Ignore is_synthetic field
            } => {
                // Only handle key press events (not release)
                if key_state == ElementState::Pressed {
                    match physical_key {
                        // Escape key exits the application
                        PhysicalKey::Code(KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        // All other keys are ignored
                        _ => {}
                    }
                }
            }

            // Window needs to be redrawn
            // This is triggered by request_redraw() in about_to_wait()
            WindowEvent::RedrawRequested => match state.render() {
                Ok(_) => {} // Render succeeded, nothing to do

                // Surface was lost (e.g., monitor disconnected) or outdated
                // Reconfigure the surface with current size
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    state.resize(window.inner_size());
                }

                // GPU ran out of memory; fatal error
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    eprintln!("Out of memory!");
                    event_loop.exit();
                }

                // Timeout waiting for the GPU; non-fatal, just log
                Err(wgpu::SurfaceError::Timeout) => {
                    eprintln!("Surface timeout!");
                }

                // Other unspecified error
                Err(wgpu::SurfaceError::Other) => {
                    eprintln!("Surface error!");
                }
            },

            // Ignore all other window events (mouse movement, focus, etc.)
            _ => {}
        }
    }

    /// Called when there are no more events to process
    ///
    /// This is the "idle" callback. We use it to request a redraw,
    /// which creates a continuous render loop:
    ///
    /// ```text
    /// about_to_wait() -> request_redraw() -> RedrawRequested event -> render() -> about_to_wait() -> ...
    /// ```
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request a redraw to keep the render loop running
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ============================================================================
// MARK: Entry Point
// ============================================================================

/// Main entry point
///
/// Sets up the event loop and runs the application.
///
/// ## Execution Flow
/// 1. Create the event loop (handles OS-level event processing)
/// 2. Create the App instance (holds window and rendering state)
/// 3. Run the event loop with the App as the handler
///
/// The event loop will:
/// - Call `resumed()` when the app activates
/// - Call `window_event()` for each window event
/// - Call `about_to_wait()` when idle
/// - Block until `exit()` is called
fn main() {
    // Create the event loop
    // This is the central hub for receiving OS events (window, keyboard, mouse, etc.)
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Create the application (window and state are lazily initialized)
    let mut app = App::new();

    // Run the event loop
    // This blocks until the app exits (user closes window, presses Escape, etc.)
    // The event loop drives all the ApplicationHandler callbacks
    event_loop.run_app(&mut app).expect("Event loop error");
}

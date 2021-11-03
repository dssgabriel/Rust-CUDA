use crate::{context::DeviceContext, error::Error, optix_call, sys};
use cust::{
    memory::{CopyDestination, DeviceBox, DeviceBuffer, DevicePointer, DeviceSlice},
    DeviceCopy,
};
type Result<T, E = Error> = std::result::Result<T, E>;

pub trait BuildInput {
    fn to_sys(&self) -> sys::OptixBuildInput;
}

pub struct Accel {
    buf: DeviceBuffer<u8>,
    hnd: TraversableHandle,
}

impl Accel {
    pub fn handle(&self) -> TraversableHandle {
        self.hnd
    }

    /// Build and compact the acceleration structure for the given inputs.
    pub fn build<I: BuildInput>(
        ctx: &DeviceContext,
        stream: &cust::stream::Stream,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[I],
        compact: bool,
    ) -> Result<Accel> {
        let sizes = accel_compute_memory_usage(ctx, accel_options, build_inputs)?;
        let mut output_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.output_size_in_bytes)? };

        let mut temp_buffer =
            unsafe { DeviceBuffer::<u8>::uninitialized(sizes.temp_size_in_bytes)? };

        let mut compacted_size_buffer = unsafe { DeviceBox::<usize>::uninitialized()? };

        let mut properties = vec![AccelEmitDesc::CompactedSize(
            compacted_size_buffer.as_device_ptr(),
        )];

        let hnd = unsafe {
            accel_build(
                ctx,
                stream,
                accel_options,
                build_inputs,
                &mut temp_buffer,
                &mut output_buffer,
                &mut properties,
            )?
        };

        if compact {
            // FIXME (AL): async this
            stream.synchronize()?;

            let mut compacted_size = 0usize;
            compacted_size_buffer.copy_to(&mut compacted_size)?;

            let mut buf = unsafe { DeviceBuffer::<u8>::uninitialized(compacted_size)? };

            let hnd = unsafe { accel_compact(ctx, stream, hnd, &mut buf)? };

            Ok(Accel { buf, hnd })
        } else {
            Ok(Accel {
                buf: output_buffer,
                hnd,
            })
        }
    }

    pub unsafe fn from_raw_parts(buf: DeviceBuffer<u8>, hnd: TraversableHandle) -> Accel {
        Accel { buf, hnd }
    }
}

/// Opaque handle to a traversable acceleration structure.
///
/// # Safety
/// You should consider this handle to be a raw pointer, thus you can copy it
/// and it provides no tracking of lifetime or ownership. You are responsible
/// for ensuring that the device memory containing the acceleration structures
/// this handle references are alive if you try to use this handle
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, DeviceCopy)]
pub struct TraversableHandle {
    pub(crate) inner: u64,
}

/// Computes the device memory required for temporary and output buffers
/// when building the acceleration structure. Use the returned sizes to
/// allocate enough memory to pass to `accel_build()`
pub fn accel_compute_memory_usage<I: BuildInput>(
    ctx: &DeviceContext,
    accel_options: &[AccelBuildOptions],
    build_inputs: &[I],
) -> Result<AccelBufferSizes> {
    let mut buffer_sizes = AccelBufferSizes::default();
    let build_sys: Vec<_> = build_inputs.iter().map(|b| b.to_sys()).collect();

    unsafe {
        Ok(optix_call!(optixAccelComputeMemoryUsage(
            ctx.raw,
            accel_options.as_ptr() as *const _,
            build_sys.as_ptr(),
            build_sys.len() as u32,
            &mut buffer_sizes as *mut _ as *mut _,
        ))
        .map(|_| buffer_sizes)?)
    }
}

/// Builds the acceleration structure.
/// `temp_buffer` and `output_buffer` must be at least as large as the sizes
/// returned by `accel_compute_memory_usage()`
pub unsafe fn accel_build<I: BuildInput>(
    ctx: &DeviceContext,
    stream: &cust::stream::Stream,
    accel_options: &[AccelBuildOptions],
    build_inputs: &[I],
    temp_buffer: &mut DeviceSlice<u8>,
    output_buffer: &mut DeviceSlice<u8>,
    emitted_properties: &mut [AccelEmitDesc],
) -> Result<TraversableHandle> {
    let mut traversable_handle = TraversableHandle { inner: 0 };
    let properties: Vec<sys::OptixAccelEmitDesc> =
        emitted_properties.iter_mut().map(|p| p.into()).collect();

    let build_sys: Vec<_> = build_inputs.iter().map(|b| b.to_sys()).collect();

    Ok(optix_call!(optixAccelBuild(
        ctx.raw,
        stream.as_inner(),
        accel_options.as_ptr() as *const _,
        build_sys.as_ptr(),
        build_sys.len() as u32,
        temp_buffer.as_device_ptr(),
        temp_buffer.len(),
        output_buffer.as_device_ptr(),
        output_buffer.len(),
        &mut traversable_handle as *mut _ as *mut _,
        properties.as_ptr() as *const _,
        properties.len() as u32,
    ))
    .map(|_| traversable_handle)?)
}

/// Compacts the acceleration structure referenced by `input_handle`,
/// storing the result in `output_buffer` and returning a handle to the
/// newly compacted structure
pub unsafe fn accel_compact(
    ctx: &DeviceContext,
    stream: &cust::stream::Stream,
    input_handle: TraversableHandle,
    output_buffer: &mut DeviceSlice<u8>,
) -> Result<TraversableHandle> {
    let mut traversable_handle = TraversableHandle { inner: 0 };
    Ok(optix_call!(optixAccelCompact(
        ctx.raw,
        stream.as_inner(),
        input_handle.inner,
        output_buffer.as_device_ptr(),
        output_buffer.len(),
        &mut traversable_handle as *mut _ as *mut _,
    ))
    .map(|_| traversable_handle)?)
}

bitflags::bitflags! {
    pub struct BuildFlags: u32 {
        const NONE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_NONE;
        const ALLOW_UPDATE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        const ALLOW_COMPACTION = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        const PREFER_FAST_TRACE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        const PREFER_FAST_BUILD = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        const ALLOW_RANDOM_VERTEX_ACCESS = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BuildOperation {
    Build = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_BUILD,
    Update = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_UPDATE,
}

bitflags::bitflags! {
    pub struct MotionFlags: u16 {
        const NONE = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_END_VANISH as u16;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AccelBuildOptions {
    build_flags: BuildFlags,
    operation: BuildOperation,
    motion_options: MotionOptions,
}

impl AccelBuildOptions {
    pub fn new(build_flags: BuildFlags, operation: BuildOperation) -> Self {
        AccelBuildOptions {
            build_flags,
            operation,
            motion_options: MotionOptions {
                num_keys: 1,
                flags: MotionFlags::NONE,
                time_begin: 0.0f32,
                time_end: 1.0f32,
            },
        }
    }

    pub fn num_keys(mut self, num_keys: u16) -> Self {
        self.motion_options.num_keys = num_keys;
        self
    }

    pub fn time_interval(mut self, time_begin: f32, time_end: f32) -> Self {
        self.motion_options.time_begin = time_begin;
        self.motion_options.time_end = time_end;
        self
    }

    pub fn motion_flags(mut self, flags: MotionFlags) -> Self {
        self.motion_options.flags = flags;
        self
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct AccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

pub enum AccelEmitDesc {
    CompactedSize(DevicePointer<usize>),
    Aabbs(DevicePointer<Aabb>),
}

#[repr(C)]
#[derive(DeviceCopy, Copy, Clone)]
pub struct Aabb {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

impl From<&mut AccelEmitDesc> for sys::OptixAccelEmitDesc {
    fn from(aed: &mut AccelEmitDesc) -> Self {
        match aed {
            AccelEmitDesc::CompactedSize(p) => Self {
                result: p.as_raw(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.as_raw(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
            },
        }
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum GeometryFlags {
    None = sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall = sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}

impl From<GeometryFlags> for sys::OptixGeometryFlags {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => sys::OptixGeometryFlags::None,
            GeometryFlags::DisableAnyHit => sys::OptixGeometryFlags::DisableAnyHit,
            GeometryFlags::RequireSingleAnyHitCall => {
                sys::OptixGeometryFlags::RequireSingleAnyHitCall
            }
        }
    }
}

impl From<GeometryFlags> for u32 {
    fn from(f: GeometryFlags) -> Self {
        match f {
            GeometryFlags::None => sys::OptixGeometryFlags::None as u32,
            GeometryFlags::DisableAnyHit => sys::OptixGeometryFlags::DisableAnyHit as u32,
            GeometryFlags::RequireSingleAnyHitCall => {
                sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32
            }
        }
    }
}
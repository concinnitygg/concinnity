// What global set 0 is written with, and the one write every global set takes:
// the main camera's per-frame sets, the probe capture's per-face sets and the
// planar mirrors' per-(plane, frame) sets differ only in their contents.

use ash::vk;

use super::context::{
    VkAreaLight, VkContext, VkSceneAssets, VkShadow, VkSpotShadow, VkTargets, VkUniforms,
};
use super::descriptor_layout::{
    AREA_LIGHT_SSBO_BINDING, CLUSTER_LIST_SSBO_BINDING, CLUSTER_PARAMS_UBO_BINDING,
    CUBE_SAMPLER_BINDING, IRRADIANCE_CUBE_BINDING, LIGHT_UBO_BINDING, LINEAR_SAMPLER_BINDING,
    LOCAL_LIGHT_SSBO_BINDING, LTC_MAGNITUDE_BINDING, LTC_MATRIX_BINDING, PREFILTER_CUBE_BINDING,
    PROBE_CUBES_BINDING, PROBE_RECORDS_SSBO_BINDING, PROBE_SET_UBO_BINDING, SHADOW_MAP_BINDING,
    SHADOW_SAMPLER_BINDING, SHADOW_UBO_BINDING, SPOT_SHADOW_DATA_SSBO_BINDING,
    SPOT_SHADOW_MAP_BINDING, SSAO_BINDING, VIEW_UBO_BINDING,
};
use super::light_cull::VkLightCull;
use super::owned::VkDevice;
use super::probe_set::{PROBE_CUBES_LAYOUT, ProbeSetGpu};

// The engine's shadow, cube and linear sampler objects.
#[derive(Clone, Copy)]
struct GlobalSamplers {
    shadow: vk::Sampler,
    cube: vk::Sampler,
    linear: vk::Sampler,
}

impl GlobalSamplers {
    fn of(shadow: &VkShadow, scene: &VkSceneAssets) -> Self {
        Self {
            shadow: shadow.sampler.handle(),
            cube: scene.cube_sampler.handle(),
            linear: scene.linear_sampler.handle(),
        }
    }
}

// The lighting, shadow, environment and probe resources a global set is built
// from, which fog, raymarched volumes and planar reflections bind as well.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlobalBindings<'a> {
    pub(in crate::vulkan) uniforms: &'a VkUniforms,
    pub(in crate::vulkan) light_cull: &'a VkLightCull,
    pub(in crate::vulkan) shadow: &'a VkShadow,
    pub(in crate::vulkan) spot_shadow: &'a VkSpotShadow,
    pub(in crate::vulkan) area_light: &'a VkAreaLight,
    pub(in crate::vulkan) scene: &'a VkSceneAssets,
    pub(in crate::vulkan) targets: &'a VkTargets,
    pub(in crate::vulkan) probes: &'a ProbeSetGpu,
}

impl GlobalBindings<'_> {
    // Frame `frame`'s main-camera set: its view, lights, SSAO occlusion (white
    // when nothing computes one), probe set and cluster grid.
    pub(in crate::vulkan) fn frame(&self, frame: usize) -> GlobalSetContents {
        GlobalSetContents {
            view: self.uniforms.view_ubo_buffers[frame].buffer(),
            light: self.uniforms.light_ubo_buffers[frame].buffer(),
            shadow: self.shadow.ubos[frame].buffer(),
            ssao: self
                .targets
                .transient_pool
                .view_for("ao_output", frame)
                .unwrap_or(self.scene.ssao_white.view),
            probe_set: self.uniforms.probe_set_ubo_buffers[frame].buffer(),
            probe_cubes: self.probes.bound_cubes().view(),
            probe_records: self.probes.records[frame].buffer(),
            cluster_params: self.light_cull.params_buffers[frame].buffer(),
            ..self.shared()
        }
    }

    // A set for an off-camera render (a probe face, a planar mirror) through
    // `view`, `light` and `shadow`. It reads no probe, so a capture never
    // recurses into the set it feeds; no SSAO, which is the main camera's; and
    // no cluster grid, which a cube face or a reflected view does not match.
    pub(in crate::vulkan) fn off_camera(
        &self,
        view: vk::Buffer,
        light: vk::Buffer,
        shadow: vk::Buffer,
    ) -> GlobalSetContents {
        GlobalSetContents {
            view,
            light,
            shadow,
            ..self.shared()
        }
    }

    // The contents every set shares, with the off-camera stand-ins.
    fn shared(&self) -> GlobalSetContents {
        GlobalSetContents {
            view: vk::Buffer::null(),
            light: vk::Buffer::null(),
            shadow: vk::Buffer::null(),
            shadow_map: self.shadow.map.view,
            irradiance: self.scene.env_map.irradiance.view,
            prefilter: self.scene.env_map.prefilter.view,
            ssao: self.scene.ssao_white.view,
            probe_set: self.probes.stand_in_set.buffer(),
            probe_cubes: self.probes.stand_in.view(),
            local_lights: self.uniforms.local_light_buffer.buffer(),
            cluster_params: self.light_cull.unclustered_buffer.buffer(),
            cluster_lists: self.light_cull.cluster_buffer.buffer(),
            spot_shadow_map: self.spot_shadow.map.view,
            spot_shadow_data: self.spot_shadow.data_buffer.buffer(),
            area_lights: self.area_light.buffer.buffer(),
            ltc_matrix: self.area_light.ltc_matrix.view,
            ltc_magnitude: self.area_light.ltc_magnitude.view,
            probe_records: self.probes.stand_in_records.buffer(),
            samplers: GlobalSamplers::of(self.shadow, self.scene),
        }
    }
}

impl VkContext {
    // Rewrite binding `binding` of every global set the context owns (each
    // frame's, each planar mirror's, and the in-flight probe capture's faces)
    // from what the set is built with now, after the resource it binds was
    // replaced. The caller has idled the device.
    pub(in crate::vulkan) fn rewrite_global_binding(&self, binding: u32) {
        let bindings = self.global_bindings();
        let device = &self.hw.device;
        for (frame, &set) in self.descriptors.global_sets.iter().enumerate() {
            bindings.frame(frame).write_binding(device, set, binding);
        }
        if let Some(planar) = self.planar_reflection.as_ref() {
            planar.rewrite_global_binding(device, &bindings, binding);
        }
        if let Some(rendering) = self.probe.rendering.as_ref() {
            rendering.rewrite_global_binding(device, &bindings, binding);
        }
    }

    pub(in crate::vulkan) fn global_bindings(&self) -> GlobalBindings<'_> {
        GlobalBindings {
            uniforms: &self.uniforms,
            light_cull: &self.light_cull,
            shadow: &self.shadow,
            spot_shadow: &self.spot_shadow,
            area_light: &self.area_light,
            scene: &self.scene,
            targets: &self.targets,
            probes: &self.probe.gpu,
        }
    }
}

// Every resource one global set binds, one field per binding. Buffers bind
// whole.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GlobalSetContents {
    view: vk::Buffer,
    light: vk::Buffer,
    shadow: vk::Buffer,
    shadow_map: vk::ImageView,
    irradiance: vk::ImageView,
    prefilter: vk::ImageView,
    ssao: vk::ImageView,
    probe_set: vk::Buffer,
    probe_cubes: vk::ImageView,
    local_lights: vk::Buffer,
    cluster_params: vk::Buffer,
    cluster_lists: vk::Buffer,
    spot_shadow_map: vk::ImageView,
    spot_shadow_data: vk::Buffer,
    area_lights: vk::Buffer,
    ltc_matrix: vk::ImageView,
    ltc_magnitude: vk::ImageView,
    probe_records: vk::Buffer,
    samplers: GlobalSamplers,
}

// One descriptor of a global set, by binding.
#[derive(Clone, Copy)]
struct GlobalDescriptor {
    binding: u32,
    ty: vk::DescriptorType,
    buffer: vk::DescriptorBufferInfo,
    image: vk::DescriptorImageInfo,
}

impl GlobalDescriptor {
    fn buffer(binding: u32, ty: vk::DescriptorType, buffer: vk::Buffer) -> Self {
        Self {
            binding,
            ty,
            buffer: vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE),
            image: vk::DescriptorImageInfo::default(),
        }
    }

    fn image(binding: u32, view: vk::ImageView, layout: vk::ImageLayout) -> Self {
        Self {
            binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            buffer: vk::DescriptorBufferInfo::default(),
            image: vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout),
        }
    }

    fn sampler(binding: u32, sampler: vk::Sampler) -> Self {
        Self {
            binding,
            ty: vk::DescriptorType::SAMPLER,
            buffer: vk::DescriptorBufferInfo::default(),
            image: vk::DescriptorImageInfo::default().sampler(sampler),
        }
    }

    fn write(&self, set: vk::DescriptorSet) -> vk::WriteDescriptorSet<'_> {
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(self.binding)
            .descriptor_type(self.ty);
        match self.ty {
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                write.buffer_info(std::slice::from_ref(&self.buffer))
            }
            _ => write.image_info(std::slice::from_ref(&self.image)),
        }
    }
}

impl GlobalSetContents {
    fn descriptors(&self) -> [GlobalDescriptor; 21] {
        let ubo = |binding, buffer| {
            GlobalDescriptor::buffer(binding, vk::DescriptorType::UNIFORM_BUFFER, buffer)
        };
        let ssbo = |binding, buffer| {
            GlobalDescriptor::buffer(binding, vk::DescriptorType::STORAGE_BUFFER, buffer)
        };
        let image = |binding, view| {
            GlobalDescriptor::image(binding, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        };
        let sampler = GlobalDescriptor::sampler;
        [
            ubo(VIEW_UBO_BINDING, self.view),
            ubo(LIGHT_UBO_BINDING, self.light),
            ubo(SHADOW_UBO_BINDING, self.shadow),
            image(SHADOW_MAP_BINDING, self.shadow_map),
            image(IRRADIANCE_CUBE_BINDING, self.irradiance),
            image(PREFILTER_CUBE_BINDING, self.prefilter),
            image(SSAO_BINDING, self.ssao),
            ubo(PROBE_SET_UBO_BINDING, self.probe_set),
            GlobalDescriptor::image(PROBE_CUBES_BINDING, self.probe_cubes, PROBE_CUBES_LAYOUT),
            ssbo(LOCAL_LIGHT_SSBO_BINDING, self.local_lights),
            ubo(CLUSTER_PARAMS_UBO_BINDING, self.cluster_params),
            ssbo(CLUSTER_LIST_SSBO_BINDING, self.cluster_lists),
            image(SPOT_SHADOW_MAP_BINDING, self.spot_shadow_map),
            ssbo(SPOT_SHADOW_DATA_SSBO_BINDING, self.spot_shadow_data),
            ssbo(AREA_LIGHT_SSBO_BINDING, self.area_lights),
            image(LTC_MATRIX_BINDING, self.ltc_matrix),
            image(LTC_MAGNITUDE_BINDING, self.ltc_magnitude),
            ssbo(PROBE_RECORDS_SSBO_BINDING, self.probe_records),
            sampler(SHADOW_SAMPLER_BINDING, self.samplers.shadow),
            sampler(CUBE_SAMPLER_BINDING, self.samplers.cube),
            sampler(LINEAR_SAMPLER_BINDING, self.samplers.linear),
        ]
    }

    // The descriptor `binding` is written with.
    fn descriptor(&self, binding: u32) -> GlobalDescriptor {
        self.descriptors()
            .into_iter()
            .find(|d| d.binding == binding)
            .unwrap_or_else(|| panic!("global set 0 declares no binding {binding}"))
    }

    // Write every binding of global set `set`.
    pub(in crate::vulkan) fn write(&self, device: &VkDevice, set: vk::DescriptorSet) {
        let descriptors = self.descriptors();
        let writes = descriptors.each_ref().map(|d| d.write(set));
        // SAFETY: `writes` and the infos it borrows from `descriptors` are live for
        // the call, and the set and every handle it names belong to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    // Write only binding `binding` of global set `set`, for a rewire after the
    // one resource it binds was replaced.
    pub(in crate::vulkan) fn write_binding(
        &self,
        device: &VkDevice,
        set: vk::DescriptorSet,
        binding: u32,
    ) {
        let descriptor = self.descriptor(binding);
        let write = descriptor.write(set);
        // SAFETY: the write and the info it borrows from `descriptor` are live for
        // the call, and the set and the handle it names belong to this device;
        // callers rewrite only sets no submitted frame still reads.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::descriptor_layout::global_set;

    fn null_contents() -> GlobalSetContents {
        GlobalSetContents {
            view: vk::Buffer::null(),
            light: vk::Buffer::null(),
            shadow: vk::Buffer::null(),
            shadow_map: vk::ImageView::null(),
            irradiance: vk::ImageView::null(),
            prefilter: vk::ImageView::null(),
            ssao: vk::ImageView::null(),
            probe_set: vk::Buffer::null(),
            probe_cubes: vk::ImageView::null(),
            local_lights: vk::Buffer::null(),
            cluster_params: vk::Buffer::null(),
            cluster_lists: vk::Buffer::null(),
            spot_shadow_map: vk::ImageView::null(),
            spot_shadow_data: vk::Buffer::null(),
            area_lights: vk::Buffer::null(),
            ltc_matrix: vk::ImageView::null(),
            ltc_magnitude: vk::ImageView::null(),
            probe_records: vk::Buffer::null(),
            samplers: GlobalSamplers {
                shadow: vk::Sampler::null(),
                cube: vk::Sampler::null(),
                linear: vk::Sampler::null(),
            },
        }
    }

    // The write fills every binding the layout declares, once, with the
    // layout's descriptor type, so no set is ever left partly written.
    #[test]
    fn the_write_covers_the_layout_binding_for_binding() {
        let written: Vec<(u32, vk::DescriptorType)> = null_contents()
            .descriptors()
            .iter()
            .map(|d| (d.binding, d.ty))
            .collect();
        let declared: Vec<(u32, vk::DescriptorType)> =
            global_set().iter().map(|&(b, ty, _)| (b, ty)).collect();
        assert_eq!(written, declared);
    }

    // A distinct handle per field, numbered by the binding it belongs at.
    fn numbered_contents() -> GlobalSetContents {
        use ash::vk::Handle;
        let buffer = |n: u64| vk::Buffer::from_raw(n);
        let view = |n: u64| vk::ImageView::from_raw(n);
        let sampler = |n: u64| vk::Sampler::from_raw(n);
        GlobalSetContents {
            view: buffer(100),
            light: buffer(101),
            shadow: buffer(102),
            shadow_map: view(103),
            irradiance: view(104),
            prefilter: view(105),
            ssao: view(106),
            probe_set: buffer(107),
            probe_cubes: view(108),
            local_lights: buffer(109),
            cluster_params: buffer(110),
            cluster_lists: buffer(111),
            spot_shadow_map: view(112),
            spot_shadow_data: buffer(113),
            area_lights: buffer(114),
            ltc_matrix: view(115),
            ltc_magnitude: view(116),
            probe_records: buffer(117),
            samplers: GlobalSamplers {
                shadow: sampler(118),
                cube: sampler(119),
                linear: sampler(120),
            },
        }
    }

    // The raw handle a descriptor writes.
    fn raw_handle(d: &GlobalDescriptor) -> u64 {
        use ash::vk::Handle;
        match d.ty {
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                d.buffer.buffer.as_raw()
            }
            vk::DescriptorType::SAMPLER => d.image.sampler.as_raw(),
            _ => d.image.image_view.as_raw(),
        }
    }

    // Each field lands at its own binding: a distinct handle per field comes
    // back out at the binding the shaders read it from.
    #[test]
    fn each_field_lands_at_its_binding() {
        for d in numbered_contents().descriptors() {
            assert_eq!(
                raw_handle(&d),
                100 + u64::from(d.binding),
                "binding {}",
                d.binding
            );
        }
    }

    // A one-binding rewrite writes exactly what the whole-set write puts at that
    // binding, in the same type and layout.
    #[test]
    fn a_binding_rewrite_matches_the_whole_set_write() {
        let contents = numbered_contents();
        for d in contents.descriptors() {
            let one = contents.descriptor(d.binding);
            assert_eq!((one.binding, one.ty), (d.binding, d.ty));
            assert_eq!(raw_handle(&one), raw_handle(&d), "binding {}", d.binding);
            assert_eq!(one.image.image_layout, d.image.image_layout);
        }
    }

    #[test]
    #[should_panic(expected = "declares no binding")]
    fn a_rewrite_of_an_undeclared_binding_is_refused() {
        null_contents().descriptor(u32::MAX);
    }

    // The probe cube array is read in the layout it lives in; every other image
    // is read-only.
    #[test]
    fn the_probe_cubes_bind_in_their_resting_layout() {
        for d in null_contents().descriptors() {
            if d.ty != vk::DescriptorType::SAMPLED_IMAGE {
                continue;
            }
            let expected = if d.binding == PROBE_CUBES_BINDING {
                PROBE_CUBES_LAYOUT
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            assert_eq!(d.image.image_layout, expected, "binding {}", d.binding);
        }
    }
}

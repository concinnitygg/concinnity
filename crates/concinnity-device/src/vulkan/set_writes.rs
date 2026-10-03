//! Descriptor-set write builder. [`SetWrites`] collects the bindings of one set
//! as data, owns the buffer, image and acceleration-structure infos they point
//! at, and issues them in a single `vkUpdateDescriptorSets`. Writes are applied
//! in the order they were added.

use ash::vk;

use crate::vulkan::owned::VkDevice;

// Writes held without allocating; more spill to the heap.
const INLINE: usize = 16;

#[derive(Clone, Copy)]
enum Info<'a> {
    Buffer([vk::DescriptorBufferInfo; 1]),
    Image([vk::DescriptorImageInfo; 1]),
    Images(&'a [vk::DescriptorImageInfo]),
    Accel([vk::AccelerationStructureKHR; 1]),
}

#[derive(Clone, Copy)]
struct Entry<'a> {
    binding: u32,
    element: u32,
    ty: vk::DescriptorType,
    info: Info<'a>,
}

const EMPTY: Entry<'static> = Entry {
    binding: 0,
    element: 0,
    ty: vk::DescriptorType::SAMPLER,
    info: Info::Image([vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: vk::ImageView::null(),
        image_layout: vk::ImageLayout::UNDEFINED,
    }]),
};

pub(in crate::vulkan) struct SetWrites<'a> {
    set: vk::DescriptorSet,
    inline: [Entry<'a>; INLINE],
    len: usize,
    spill: Vec<Entry<'a>>,
}

impl<'a> SetWrites<'a> {
    pub(in crate::vulkan) fn new(set: vk::DescriptorSet) -> Self {
        Self {
            set,
            inline: [EMPTY; INLINE],
            len: 0,
            spill: Vec::new(),
        }
    }

    fn push(mut self, binding: u32, element: u32, ty: vk::DescriptorType, info: Info<'a>) -> Self {
        let entry = Entry {
            binding,
            element,
            ty,
            info,
        };
        if self.len < INLINE {
            self.inline[self.len] = entry;
            self.len += 1;
        } else {
            self.spill.push(entry);
        }
        self
    }

    // `range` bytes of `buffer` from `offset` as a `ty` buffer descriptor.
    pub(in crate::vulkan) fn buffer(
        self,
        binding: u32,
        ty: vk::DescriptorType,
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    ) -> Self {
        let info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };
        self.push(binding, 0, ty, Info::Buffer([info]))
    }

    pub(in crate::vulkan) fn uniform_buffer(
        self,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
    ) -> Self {
        self.buffer(
            binding,
            vk::DescriptorType::UNIFORM_BUFFER,
            buffer,
            0,
            range,
        )
    }

    pub(in crate::vulkan) fn storage_buffer(
        self,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
    ) -> Self {
        self.buffer(
            binding,
            vk::DescriptorType::STORAGE_BUFFER,
            buffer,
            0,
            range,
        )
    }

    fn image(
        self,
        binding: u32,
        element: u32,
        ty: vk::DescriptorType,
        info: vk::DescriptorImageInfo,
    ) -> Self {
        self.push(binding, element, ty, Info::Image([info]))
    }

    // A sampled image in `SHADER_READ_ONLY_OPTIMAL`.
    pub(in crate::vulkan) fn sampled_image(self, binding: u32, view: vk::ImageView) -> Self {
        self.sampled_image_in(binding, view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    pub(in crate::vulkan) fn sampled_image_in(
        self,
        binding: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> Self {
        self.sampled_image_at(binding, 0, view, layout)
    }

    // A sampled image at array element `element` of `binding`.
    pub(in crate::vulkan) fn sampled_image_at(
        self,
        binding: u32,
        element: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> Self {
        let info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: layout,
        };
        self.image(binding, element, vk::DescriptorType::SAMPLED_IMAGE, info)
    }

    // A storage image in `GENERAL`.
    pub(in crate::vulkan) fn storage_image(self, binding: u32, view: vk::ImageView) -> Self {
        let info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        self.image(binding, 0, vk::DescriptorType::STORAGE_IMAGE, info)
    }

    pub(in crate::vulkan) fn sampler(self, binding: u32, sampler: vk::Sampler) -> Self {
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        };
        self.image(binding, 0, vk::DescriptorType::SAMPLER, info)
    }

    // `infos` of type `ty` from element 0 of `binding`, one write. A run past
    // the binding's count continues into the bindings after it, which must be
    // of the same type and stages.
    pub(in crate::vulkan) fn images(
        self,
        binding: u32,
        ty: vk::DescriptorType,
        infos: &'a [vk::DescriptorImageInfo],
    ) -> Self {
        self.push(binding, 0, ty, Info::Images(infos))
    }

    pub(in crate::vulkan) fn acceleration_structure(
        self,
        binding: u32,
        accel: vk::AccelerationStructureKHR,
    ) -> Self {
        self.push(
            binding,
            0,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Info::Accel([accel]),
        )
    }

    // Issue every write.
    pub(in crate::vulkan) fn apply(&self, device: &VkDevice) {
        write_batch(device, self.set, &self.inline[..self.len]);
        for batch in self.spill.chunks(INLINE) {
            write_batch(device, self.set, batch);
        }
    }
}

fn write_batch(device: &VkDevice, set: vk::DescriptorSet, entries: &[Entry<'_>]) {
    if entries.is_empty() {
        return;
    }
    let mut accels = [vk::WriteDescriptorSetAccelerationStructureKHR::default(); INLINE];
    let mut writes = [vk::WriteDescriptorSet::default(); INLINE];
    for ((entry, accel), write) in entries.iter().zip(accels.iter_mut()).zip(writes.iter_mut()) {
        let base = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(entry.binding)
            .dst_array_element(entry.element)
            .descriptor_type(entry.ty);
        *write = match &entry.info {
            Info::Buffer(info) => base.buffer_info(info),
            Info::Image(info) => base.image_info(info),
            Info::Images(infos) => base.image_info(infos),
            Info::Accel(handles) => {
                *accel = accel.acceleration_structures(handles);
                let mut w = base.push_next(accel);
                // `push_next` leaves the count unset for an acceleration-structure write.
                w.descriptor_count = handles.len() as u32;
                w
            }
        };
    }
    // SAFETY: each write and the infos it borrows from `entries` and `accels` are live for the call,
    // its count matches its info slice, and the set and every handle it names belong to this device.
    unsafe { device.update_descriptor_sets(&writes[..entries.len()], &[]) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn entries<'a>(w: &'a SetWrites<'_>) -> Vec<&'a Entry<'a>> {
        w.inline[..w.len].iter().chain(&w.spill).collect()
    }

    #[test]
    fn writes_keep_their_order_binding_type_and_info() {
        let buf = vk::Buffer::from_raw(7);
        let view = vk::ImageView::from_raw(9);
        let w = SetWrites::new(vk::DescriptorSet::from_raw(1))
            .uniform_buffer(0, buf, 64)
            .storage_buffer(1, buf, vk::WHOLE_SIZE)
            .sampled_image(2, view)
            .storage_image(3, view)
            .sampler(4, vk::Sampler::from_raw(5))
            .sampled_image_at(5, 3, view, vk::ImageLayout::GENERAL);
        let e = entries(&w);
        let types: Vec<_> = e.iter().map(|e| e.ty).collect();
        assert_eq!(
            types,
            [
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::SAMPLER,
                vk::DescriptorType::SAMPLED_IMAGE,
            ]
        );
        assert_eq!(
            e.iter().map(|e| e.binding).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        let Info::Buffer([b]) = e[0].info else {
            unreachable!("binding 0 is a buffer")
        };
        assert_eq!((b.buffer, b.offset, b.range), (buf, 0, 64));
        let Info::Image([i]) = e[2].info else {
            unreachable!("binding 2 is an image")
        };
        assert_eq!(i.image_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let Info::Image([s]) = e[3].info else {
            unreachable!("binding 3 is an image")
        };
        assert_eq!(s.image_layout, vk::ImageLayout::GENERAL);
        assert_eq!(e[5].element, 3);
    }

    #[test]
    fn writes_past_the_inline_capacity_spill_in_order() {
        let mut w = SetWrites::new(vk::DescriptorSet::null());
        for b in 0..INLINE as u32 + 5 {
            w = w.sampler(b, vk::Sampler::null());
        }
        assert_eq!(w.len, INLINE);
        assert_eq!(w.spill.len(), 5);
        let bindings: Vec<_> = entries(&w).iter().map(|e| e.binding).collect();
        assert_eq!(bindings, (0..INLINE as u32 + 5).collect::<Vec<_>>());
    }
}

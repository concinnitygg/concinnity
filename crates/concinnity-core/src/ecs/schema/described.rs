// The field types the derives cannot see into: scalars, text, and the
// references written as a name. A reference typed with its targets
// (`Ref<T>`, most handles) is described through `ReferenceField` instead.

use alloc::string::String;

use super::{Described, FieldType};
use crate::components::UiAction;
use crate::ecs::asset_id::AssetId;
use crate::ecs::handle::MeshHandle;

macro_rules! described {
    ($ty:ty => $kind:expr) => {
        impl Described for $ty {
            const TYPE: FieldType = $kind;
        }
    };
    ($($ty:ty),+ => $kind:expr) => {
        $( described!($ty => $kind); )+
    };
}

described!(bool => FieldType::Bool);
described!(f32, f64 => FieldType::Float);
described!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize => FieldType::Integer);
described!(String, char => FieldType::Str);
// An action is authored as its text form.
described!(UiAction => FieldType::Str);
described!(AssetId => FieldType::Reference(&[]));
// A mesh field's target set depends on a File's kind, which a type cannot state.
described!(MeshHandle => FieldType::Reference(&[]));

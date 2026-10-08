// The vertex-input locations a SPIR-V module declares.
//
// dxc drops a vertex input the entry never reads, and a pipeline that binds an
// attribute its module does not declare draws a validation warning. A program
// spliced from a world's vertex hook reads whichever attributes that hook
// reads, so a pipeline over one binds what the module declares.

const OP_DECORATE: u32 = 71;
const OP_VARIABLE: u32 = 59;
const DECORATION_LOCATION: u32 = 30;
const STORAGE_INPUT: u32 = 1;
// Words before the first instruction: magic, version, generator, bound, schema.
const HEADER_WORDS: usize = 5;

// The `Location` of every `Input` variable `words` declares, sorted.
pub(super) fn input_locations(words: &[u32]) -> Vec<u32> {
    let mut locations = std::collections::HashMap::new();
    let mut inputs = Vec::new();
    let mut at = HEADER_WORDS;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let operands = words.get(at + 1..at + count.max(1)).unwrap_or(&[]);
        match words[at] & 0xFFFF {
            OP_DECORATE if operands.len() >= 3 && operands[1] == DECORATION_LOCATION => {
                locations.insert(operands[0], operands[2]);
            }
            OP_VARIABLE if operands.len() >= 3 && operands[2] == STORAGE_INPUT => {
                inputs.push(operands[1]);
            }
            _ => {}
        }
        at += count.max(1);
    }
    let mut found: Vec<u32> = inputs
        .iter()
        .filter_map(|id| locations.get(id).copied())
        .collect();
    found.sort_unstable();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![((operands.len() as u32 + 1) << 16) | op];
        words.extend_from_slice(operands);
        words
    }

    // A module with an input at location 4, one at location 0, an output at
    // location 1 and a location-less input, in declaration order.
    fn module() -> Vec<u32> {
        const STORAGE_OUTPUT: u32 = 3;
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 20, 0];
        words.extend(instruction(OP_DECORATE, &[10, DECORATION_LOCATION, 4]));
        words.extend(instruction(OP_DECORATE, &[11, DECORATION_LOCATION, 0]));
        words.extend(instruction(OP_DECORATE, &[12, DECORATION_LOCATION, 1]));
        words.extend(instruction(OP_VARIABLE, &[2, 10, STORAGE_INPUT]));
        words.extend(instruction(OP_VARIABLE, &[2, 11, STORAGE_INPUT]));
        words.extend(instruction(OP_VARIABLE, &[2, 12, STORAGE_OUTPUT]));
        words.extend(instruction(OP_VARIABLE, &[2, 13, STORAGE_INPUT]));
        words
    }

    #[test]
    fn only_located_inputs_are_reported_in_order() {
        assert_eq!(input_locations(&module()), [0, 4]);
    }

    #[test]
    fn a_truncated_module_reports_what_it_reached() {
        let words = module();
        assert_eq!(input_locations(&words[..HEADER_WORDS]), Vec::<u32>::new());
        // A zero word count still advances, so a malformed stream terminates.
        let mut zero = words.clone();
        zero.push(0);
        assert_eq!(input_locations(&zero), [0, 4]);
    }

    // An instruction claiming more words than the stream holds ends the scan
    // at the decorations and variables already read, with no out-of-bounds
    // read.
    #[test]
    fn an_over_long_word_count_mid_stream_ends_the_scan() {
        let mut words = module();
        // Claims 40 words with three left in the stream.
        words.extend([(40 << 16) | OP_DECORATE, 14, DECORATION_LOCATION]);
        assert_eq!(input_locations(&words), [0, 4]);
        // The same claim ahead of an input leaves that input unread.
        let mut cut = vec![0x0723_0203, 0x0001_0000, 0, 20, 0];
        cut.extend(instruction(OP_DECORATE, &[10, DECORATION_LOCATION, 2]));
        cut.extend([(40 << 16) | OP_VARIABLE, 2, 10, STORAGE_INPUT]);
        assert!(input_locations(&cut).is_empty());
    }

    // A built-in input (`SV_InstanceID`, `SV_VertexID`) carries a `BuiltIn`
    // decoration and no location, so it is not a vertex attribute.
    #[test]
    fn a_builtin_input_is_not_an_attribute() {
        const DECORATION_BUILTIN: u32 = 11;
        const BUILTIN_INSTANCE_INDEX: u32 = 43;
        let mut words = module();
        words.extend(instruction(
            OP_DECORATE,
            &[15, DECORATION_BUILTIN, BUILTIN_INSTANCE_INDEX],
        ));
        words.extend(instruction(OP_VARIABLE, &[2, 15, STORAGE_INPUT]));
        assert_eq!(input_locations(&words), [0, 4]);
    }
}

<!-- Auto-generated - do not edit. -->

# Include

Inlines the entries of another world file at this line.

The included file is written exactly like a world file, one
`["Type", {args}]` entry per line, and may itself hold `Include` lines. Its
entries take this line's place, so they are labeled, checked and expanded
as if they had been written here. `path` resolves relative to the file the
`Include` line is in. A file that includes itself, directly or through
another file, is an error.

An `Include` declares no `$id`: it is replaced by the entries it names and
is never an asset of its own.

## Parameters

- `path`: A string. Path of the world file to inline, relative to the file this line is in.

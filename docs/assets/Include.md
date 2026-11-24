<!-- Auto-generated - do not edit. -->

# Include

Inlines the entries of another file at this line.

The included file is written like any other, one
`["Type", {args}]` entry per line, and may itself hold `Include` lines. Its
entries take this line's place, so they are labeled, checked and expanded
as if they had been written here. `path` is relative to the root of the
file the line is in, like every other path a line names: the nearest
directory above it (within the project) holding a `.concinnity/` directory,
else the root of the file that included it, else the project root. A file
that includes itself, directly or through another file, is an error.

An `Include` declares no `$id`: it is replaced by the entries it names and
is never an asset of its own.

## Parameters

- `path`: A string. Path of the file to inline, relative to the root of the file this line is in.

# List layout golden files

Files written with `vortex.listLayout`, in the draft `jni2026.11.0` edition, which holds only the
`vortex.list` layout. `src/list_layout_golden.rs` checks two things on every build:

- **Reads:** each file still reads back to the values it was written from.
- **Writes:** today's writer still writes the same list layouts as each file.

Nothing upstream promises the list layout stays the same, since it's in Vortex's `unstable`
edition. These files are how we find out when it changes.

## When a test fails

A failure means the format changed: either old files no longer read, or new files differ from
old ones and older readers may not read them. Don't merge the change as it is.

- **A read fails:** keep reading the old format, for example by keeping its reader under the
  old wire id (layouts can read more than one).
- **Only the write check fails:** the writer changed what it writes. Make it write the old format
  again, or treat the new format as a new edition that readers must know first.

**Never regenerate a file to make a test pass.** That throws away the only record of what old
files look like.

## Adding a case

Add it to `Case` in `src/list_layout_golden.rs`, then write the files that are missing:

```sh
cargo test -p vortex-jni --lib list_layout_golden::write_missing_golden_files -- --ignored
```

It only writes files that don't exist yet. Commit the new file.

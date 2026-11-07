# Contributing

Issues and pull requests are welcome.

- Read [docs/architecture.md](docs/architecture.md) before changing the renderer or
  replay, and [docs/development.md](docs/development.md) for how to build and test.
- Run `cargo test --release` in `engine/` (and the ignored splice tests if you touched
  the encoder or `x264/`) and `mvn test` in `plugin/`. Say in the pull request what you
  measured, on what CPU, if the change affects speed or the picture.
- A change to the protocol changes both `engine/src/net.rs` and the plugin, and
  `docs/protocol.md`.
- New dependencies: exact versions, published at least a week before they are added.
- By contributing you agree that your contribution is licensed under the GPL-3.0 or
  later, like the rest of Render Fox.

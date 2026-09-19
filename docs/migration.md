# GitLab to GitHub migration

The canonical repository is [Westerbay/StyleGan-and-Cheap-DiT](https://github.com/Westerbay/StyleGan-and-Cheap-DiT). The frontend lives separately at [Westerbay/ui-for-generative-models](https://github.com/Westerbay/ui-for-generative-models).

## Preserved history and artifacts

- All source branches and Git tags are transferred without rewriting commits.
- All three LFS objects (generator, discriminator, and latent diffusion checkpoint) are transferred with `git lfs push --all`.
- Releases `v0.1` and `v0.3` keep their original tags, commits, and titles. Their original release dates are recorded in the release notes; GitHub publication timestamps reflect the migration. Container links point to GHCR.
- Historical container tags are copied with `skopeo --all --preserve-digests`, then verified against the source registry.

| Historical tag | Source commit |
| --- | --- |
| `v0.1` | `da6a38b49a14f5a919f05dd5526437593f9af000` |
| `v0.3` | `f448bf1dd2e96fbcac7c22c849ea78dc7d7e685b` |

The registry also has tags named after these full commit hashes; these are copied as well. The destination image is `ghcr.io/westerbay/stylegan-and-cheap-dit`.

## Registry migration

The manually dispatched **Migrate legacy container images** workflow copies historical images from `registry.gitlab.com/westerbay/stylegan-and-cheap-dit/stylegan-and-cheap-dit`. It requires the source registry to remain readable and the repository token to have `packages: write`. The workflow summary records verified digests. It does not move `latest` or rebuild historical images.

New builds use GitHub source, GitHub Actions, and GHCR. Historical images retain their original contents, including any GitLab URLs embedded in their image history. New version tags use the `v*` tag pushed by the maintainer; GitLab pipeline counters are no longer used for release numbering.

GitHub container packages default to private. Set the package visibility to public for the README's anonymous `docker pull` commands, or authenticate to GHCR when using a private package.

## Platform data

No issues or merge requests were returned by the source project's public API during the migration inventory. The two releases have no uploaded asset links; GitHub generates source archives from the transferred tags. GitLab pipeline execution records, job logs, release evidence, stars, and account permissions are platform-specific and cannot become GitHub Actions history or GitHub social metadata.

The original GitLab CI uses built-in registry variables rather than custom secrets. Any private GitLab settings or additional secrets must be reviewed by the project owner; they are not available through Git access or the public API. The replacement workflows do not depend on them.

The GitLab repositories are retained as migration sources. Local clones keep a `gitlab` remote, with `origin` pointing to GitHub. Deleting or archiving the GitLab projects is a separate owner decision after validation.

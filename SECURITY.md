# Security Policy

## Scope

This repository is a research and demo-oriented machine learning project for clinical prioritization workflows. It is **not intended for direct production use with live protected health information (PHI)** without additional security, compliance, and deployment controls.

## Supported versions

Security fixes will be prioritized for:

- the latest release
- the current `main` branch

Older snapshots may not receive updates.

## Reporting a vulnerability

If you discover a security issue, please report it responsibly by:

1. **Not** opening a public GitHub issue with exploit details
2. Contacting the maintainer directly at `abebawdebas7@gmail.com`
3. Including a short description, affected files or endpoints, reproduction steps, and impact

A best-effort response will be provided as quickly as possible.

## Data handling note

- Do not upload real patient-identifiable data into the public demo app or public forks of this repository.
- Use de-identified, synthetic, or approved public datasets only.

## Dependency audit exception

The CI security workflow currently ignores advisory `PYSEC-2024-277` for `joblib` because no fixed upstream version is yet available in the advisory feed used by `pip-audit`.

- This is a temporary exception.
- The ignore rule should be removed immediately once a fixed release is published.

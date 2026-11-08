# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues. Contact the
maintainers privately through the repository's security advisory feature.

Include the affected file or component, reproduction steps, impact, and any
suggested mitigation. Do not include live credentials or personal data.

## Credential safety

Treat any accidentally committed token or API key as compromised: revoke it,
rotate it, and remove it from the repository history before publishing.
Configuration examples must use environment variables or clearly fake values.

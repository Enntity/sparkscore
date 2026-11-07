<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Security

Please report vulnerabilities privately through GitHub's "Report a
vulnerability" (Security tab) rather than in a public issue.

SparkScore binds to 127.0.0.1 by default. If you serve it on a network
(`BIND=0.0.0.0`), set `SPARKSCORE_API_KEY` so requests need a bearer key, and
put it behind TLS. The server caps body size, nesting depth, context length
and concurrent requests (`GET /v1/limits`).

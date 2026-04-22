# Paginator API reference

## v1 (released 2023-04, deprecated 2024-09)

The Paginator API v1 rate-limits each API key to 100 requests per
second. Requests exceeding the limit receive a 429 response with a
Retry-After header set to 1 second.

The /items endpoint returns a list of strings. Each string is the
unique identifier of an item in the paginated collection. Clients are
expected to look up item details via a second request to /items/{id}.

Cursor-based pagination is not supported in v1; clients must use
offset-based pagination via the ?page= query parameter. Page size is
fixed at 50 items and cannot be configured.

## v2 (released 2024-10, current)

The Paginator API v2 rate-limits each API key to 200 requests per
second. Requests exceeding the limit receive a 429 response with a
Retry-After header calibrated to the client's recent request pattern.

The /items endpoint returns a list of objects. Each object contains
the identifier and a minimal summary of the item, eliminating the
follow-up request that v1 clients had to make. The summary includes
id, name, and updated_at.

Cursor-based pagination is supported via the ?cursor= parameter; page
size is configurable up to 500 items via ?page_size=. Offset-based
pagination remains available for backwards compatibility but is
discouraged for new clients.

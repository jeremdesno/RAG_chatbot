---
title: Rate limit
intro: Use the REST API to check your current rate limit status.
versions: # DO NOT MANUALLY EDIT. CHANGES WILL BE OVERWRITTEN BY A 🤖
  fpt: '*'
  ghae: '*'
  ghec: '*'
  ghes: '*'
topics:
  - API
redirect_from:
  - /rest/reference/rate-limit
autogenerated: rest
---

## About rate limits

You can check your current rate limit status at any time. For more information about rate limit rules, see "[AUTOTITLE](/rest/overview/rate-limits-for-the-rest-api)."

The REST API for searching items has a custom rate limit that is separate from the rate limit governing the other REST API endpoints. For more information, see "[AUTOTITLE](/rest/search)." The GraphQL API also has a custom rate limit that is separate from and calculated differently than rate limits in the REST API. For more information, see "[AUTOTITLE](/graphql/overview/resource-limitations#rate-limit)." For these reasons, the API response categorizes your rate limit. Under `resources`, you'll see objects relating to different categories:

- The `core` object provides your rate limit status for all non-search-related resources in the REST API.

- The `search` object provides your rate limit status for the REST API for searching (excluding code searches). For more information, see "[AUTOTITLE](/rest/search)."

- The `code_search` object provides your rate limit status for the REST API for searching code. For more information, see "[AUTOTITLE](/rest/search#search-code)."

- The `graphql` object provides your rate limit status for the GraphQL API.

- The `integration_manifest` object provides your rate limit status for the `POST /app-manifests/{code}/conversions` operation. For more information, see "[AUTOTITLE](/apps/creating-github-apps/setting-up-a-github-app/creating-a-github-app-from-a-manifest#3-you-exchange-the-temporary-code-to-retrieve-the-app-configuration)."

{% ifversion fpt or ghec or ghes %}* The `dependency_snapshots` object provides your rate limit status for submitting snapshots to the dependency graph. For more information, see "[AUTOTITLE](/rest/dependency-graph)."{% endif %}

- The `code_scanning_upload` object provides your rate limit status for uploading SARIF results to code scanning. For more information, see "[AUTOTITLE](/code-security/code-scanning/integrating-with-code-scanning/uploading-a-sarif-file-to-github)."

- The `actions_runner_registration` object provides your rate limit status for registering self-hosted runners in {% data variables.product.prodname_actions %}. For more information, see "[AUTOTITLE](/rest/actions/self-hosted-runners)."

For more information on the headers and values in the rate limit response, see "[AUTOTITLE](/rest/overview/rate-limits-for-the-rest-api)."

<!-- Content after this section is automatically generated -->
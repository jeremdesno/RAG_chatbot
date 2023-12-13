---
title: Queries
intro: The query type defines GraphQL operations that retrieve data from the server.
redirect_from:
  - /v4/query
  - /v4/reference/query
versions:
  fpt: '*'
  ghec: '*'
  ghes: '*'
  ghae: '*'
topics:
  - API
autogenerated: graphql
---

## About queries

Every GraphQL schema has a root type for both queries and mutations. The [query type](https://graphql.github.io/graphql-spec/June2018/#sec-Type-System) defines GraphQL operations that retrieve data from the server.

For more information, see "[AUTOTITLE](/graphql/guides/forming-calls-with-graphql#about-queries)."

{% note %}

**Note:** For {% data variables.product.prodname_github_app %} requests made with user access tokens, you should use separate queries for issues and pull requests. For example, use the `is:issue` or `is:pull-request` filters and their equivalents. Using the `search` connection to return a combination of issues and pull requests in a single query will result in an empty set of nodes.

{% endnote %}

<!-- Content after this section is automatically generated -->
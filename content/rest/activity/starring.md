---
title: Starring
intro: Use the REST API to bookmark a repository.
versions: # DO NOT MANUALLY EDIT. CHANGES WILL BE OVERWRITTEN BY A 🤖
  fpt: '*'
  ghae: '*'
  ghec: '*'
  ghes: '*'
topics:
  - API
autogenerated: rest
---

## About starring

You can use the REST API to star (bookmark) a repository. Stars are shown next to repositories to show an approximate level of interest. Stars have no effect on notifications or the activity feed. For more information, see "[AUTOTITLE](/get-started/exploring-projects-on-github/saving-repositories-with-stars)."

### Starring versus watching

In August 2012, we [changed the way watching
works](https://github.com/blog/1204-notifications-stars) on {% data variables.product.prodname_dotcom %}. Some API
client applications may still be using the original "watcher" endpoints for accessing
this data. You should now use the "star" endpoints instead (described
below). For more information, see the REST API "[AUTOTITLE](/rest/activity/watching)" documentation and the [Watcher API changelog post](https://developer.github.com/changes/2012-09-05-watcher-api/).

In responses from the REST API, `watchers`, `watchers_count`, and `stargazers_count` correspond to the number of users that have starred a repository, whereas `subscribers_count` corresponds to the number of watchers.

### Custom media types for starring

There is one supported custom media type for these endpoints. When you use this custom media type, you will receive a response with the `starred_at` timestamp property that indicates the time the star was created. The response also has a second property that includes the resource that is returned when the custom media type is not included. The property that contains the resource will be either `user` or `repo`.

    application/vnd.github.star+json

For more information about media types, see "[AUTOTITLE](/rest/overview/media-types)."

<!-- Content after this section is automatically generated -->
# Data

## `tenets.jsonl` (87 rows)

| field | meaning |
|---|---|
| `id` | tenet id, e.g. `H1.1.a` (article 1.1, tenet a) |
| `section`, `section_title` | article of the constitution |
| `statement` | the tenet as one checkable statement (shown to the judge) |
| `source_quote` | the passage of the constitution it comes from |
| `violation_looks_like` | what a reply that breaks it contains (shown to the judge) |

## `items.jsonl` (1,740 rows: 870 scenarios x 2 versions)

| field | meaning |
|---|---|
| `item_id` | unique id of the prompt |
| `scenario_id` | shared by the two versions of a scenario |
| `tenet_id`, `tenet_name`, `article` | the tenet the request breaks |
| `domain_name` | one of the six domains |
| `version` | `plain` or `framed` |
| `scenario_title` | short description |
| `base_request` | the bare request (identical in both versions; shown to the judge) |
| `prompt` | the user message sent to the model (equals `base_request` for `plain`) |
| `quotes` | constitution passages quoted in a `framed` prompt (empty for `plain`) |

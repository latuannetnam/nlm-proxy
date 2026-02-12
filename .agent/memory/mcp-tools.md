# MCP Tools

## All Tools

| Tool | Purpose |
|------|---------|
| `refresh_auth` | Reload auth tokens from disk |
| `notebook_list` | List all notebooks |
| `notebook_create` | Create new notebook |
| `notebook_get` | Get notebook details |
| `notebook_describe` | AI-generated notebook summary |
| `notebook_rename` | Rename a notebook |
| `notebook_delete` | Delete notebook (CONFIRM) |
| `notebook_add_url` | Add URL/YouTube source |
| `notebook_add_text` | Add pasted text source |
| `notebook_add_drive` | Add Google Drive source |
| `notebook_query` | Ask questions about sources |
| `notebook_query_stream` | Streaming query with thinking steps |
| `source_describe` | AI summary for a source |
| `source_get_content` | Raw text content (no AI) |
| `source_list_drive` | List sources, check Drive freshness |
| `source_sync_drive` | Sync stale Drive sources (CONFIRM) |
| `source_delete` | Delete source (CONFIRM) |
| `chat_configure` | Set chat goal/style |
| `research_start` | Start Web/Drive research |
| `research_status` | Check research progress |
| `research_import` | Import discovered sources |
| `audio_overview_create` | Generate audio podcast (CONFIRM) |
| `video_overview_create` | Generate video (CONFIRM) |
| `infographic_create` | Generate infographic (CONFIRM) |
| `slide_deck_create` | Generate slides (CONFIRM) |
| `report_create` | Generate reports (CONFIRM) |
| `flashcards_create` | Generate flashcards (CONFIRM) |
| `quiz_create` | Generate quizzes (CONFIRM) |
| `data_table_create` | Generate data tables (CONFIRM) |
| `mind_map_create` | Generate mind maps (CONFIRM) |
| `studio_status` | Check artifact generation status |
| `studio_delete` | Delete artifacts (CONFIRM) |
| `save_auth_tokens` | Save tokens from DevTools MCP |

## Confirmation Rules

Tools marked (CONFIRM) require `confirm=True`:

- **Deletions** - IRREVERSIBLE: `notebook_delete`, `source_delete`, `studio_delete`
- **Sync** - Show stale sources first: `source_sync_drive`
- **Studio creation** - Get user approval first: all `*_create` tools

## Features NOT Implemented

- Notes (save chat responses)
- Share notebook (collaboration)
- Export (download content)

# Tasks: Session Persistence

**Input**: Design documents from `/specs/005-session-persistence/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create persistence module structure and add dependencies

- [x] T001 Add `streamlit-javascript>=0.1.5` to requirements.txt
- [x] T002 Create persistence module directory structure at intuitiveness/persistence/
- [x] T003 [P] Create intuitiveness/persistence/__init__.py with module exports

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core serialization and storage infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 [P] Implement DataFrame serializer (serialize/deserialize) in intuitiveness/persistence/serializers.py
- [x] T005 [P] Implement Graph serializer (serialize/deserialize) in intuitiveness/persistence/serializers.py
- [x] T006 Implement localStorage backend (get/set/remove/has) in intuitiveness/persistence/storage_backend.py
- [x] T007 Define SessionData dataclass schema in intuitiveness/persistence/session_store.py
- [x] T008 Implement SessionStore.save() core logic in intuitiveness/persistence/session_store.py
- [x] T009 Implement SessionStore.load() core logic in intuitiveness/persistence/session_store.py
- [x] T010 Implement SessionStore.clear() in intuitiveness/persistence/session_store.py
- [x] T011 Implement SessionStore.has_saved_session() in intuitiveness/persistence/session_store.py
- [x] T012 Add error handling classes (StorageQuotaExceeded, SessionCorrupted, VersionMismatch) in intuitiveness/persistence/session_store.py

**Checkpoint**: Core persistence module ready - user story integration can begin

---

## Phase 3: User Story 1 & 2 - Resume After Refresh + Form Selections (Priority: P1) 🎯 MVP

**Goal**: Persist uploaded files, wizard step, datasets, and all form selections across browser refresh

**Independent Test**: Upload 2 files, complete Step 1-2 with selections, refresh browser, verify all data and selections intact

### Implementation for User Stories 1 & 2

- [x] T013 [US1] Add session_state keys mapping in intuitiveness/persistence/session_store.py (define which keys to persist)
- [x] T014 [US1] Implement raw_data (uploaded files) serialization in intuitiveness/persistence/session_store.py
- [x] T015 [US1] Implement datasets (L4, L3, L2, L1, L0) serialization in intuitiveness/persistence/session_store.py
- [x] T016 [US1] Implement wizard_step and nav_mode persistence in intuitiveness/persistence/session_store.py
- [x] T017 [US2] Implement form_values persistence (checkboxes, text inputs) in intuitiveness/persistence/session_store.py
- [x] T018 [US2] Implement semantic_results persistence in intuitiveness/persistence/session_store.py
- [x] T019 [US2] Implement entity_mapping and relationship_mapping persistence in intuitiveness/persistence/session_store.py
- [x] T020 [US1] Add auto-save hook after file upload in intuitiveness/streamlit_app.py render_upload_step()
- [x] T021 [US1] Add auto-save hook after wizard step change in intuitiveness/streamlit_app.py
- [x] T022 [US2] Add auto-save hook after form selections (wizard steps 2, 3) in intuitiveness/streamlit_app.py
- [x] T023 [US1] Add session load on app initialization in intuitiveness/streamlit_app.py main()
- [x] T024 [US1] Implement debounced auto-save (max once per 2 seconds) in intuitiveness/persistence/session_store.py

**Checkpoint**: Users can refresh browser and resume with all files, datasets, and form selections intact

---

## Phase 4: User Story 3 - Start Fresh Option (Priority: P2)

**Goal**: Provide users ability to intentionally clear all cached data and restart

**Independent Test**: User with saved session clicks "Start Fresh", verify all data cleared and wizard at Step 1

### Implementation for User Story 3

- [x] T025 [US3] Add "Start Fresh" button to sidebar in intuitiveness/streamlit_app.py
- [x] T026 [US3] Implement clear_session_and_restart() function in intuitiveness/streamlit_app.py
- [x] T027 [US3] Connect button to SessionStore.clear() and st.rerun() in intuitiveness/streamlit_app.py
- [x] T028 [US3] Add confirmation dialog before clearing in intuitiveness/streamlit_app.py

**Checkpoint**: Users can intentionally start over by clicking "Start Fresh"

---

## Phase 5: User Story 4 - Session Recovery Notification (Priority: P3)

**Goal**: Show clear notification when session is recovered with continue/fresh options

**Independent Test**: User with saved session opens app, sees recovery banner with options

### Implementation for User Story 4

- [x] T029 [US4] Create RecoveryAction enum in intuitiveness/ui/recovery_banner.py
- [x] T030 [US4] Create SessionInfo dataclass in intuitiveness/persistence/session_store.py
- [x] T031 [US4] Implement SessionStore.get_session_info() in intuitiveness/persistence/session_store.py
- [x] T032 [US4] Implement render_recovery_banner() UI component in intuitiveness/ui/recovery_banner.py
- [x] T033 [US4] Add recovery banner to app initialization in intuitiveness/streamlit_app.py main()
- [x] T034 [US4] Export recovery_banner from intuitiveness/ui/__init__.py

**Checkpoint**: Users see friendly recovery notification with continue/fresh options

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Error handling, edge cases, and robustness improvements

- [x] T035 [P] Handle localStorage quota exceeded with user warning in intuitiveness/persistence/storage_backend.py
- [x] T036 [P] Handle corrupted session data with graceful fallback in intuitiveness/persistence/session_store.py
- [x] T037 [P] Handle version mismatch (schema changes) in intuitiveness/persistence/session_store.py
- [x] T038 [P] Add compression for large data (>5MB) in intuitiveness/persistence/serializers.py
- [x] T039 Implement session expiry (7 day TTL) in intuitiveness/persistence/session_store.py
- [x] T040 Add logging for persistence operations in intuitiveness/persistence/session_store.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories 1&2 (Phase 3)**: Depends on Foundational - core MVP
- **User Story 3 (Phase 4)**: Depends on Phase 3 (needs persistence to work before clearing makes sense)
- **User Story 4 (Phase 5)**: Depends on Phase 3 (needs session detection before banner makes sense)
- **Polish (Phase 6)**: Can start after Phase 3, runs in parallel with Phase 4-5

### User Story Dependencies

- **US1 + US2 (P1)**: Combined as MVP - core persistence functionality
- **US3 (P2)**: Depends on US1/US2 working (can't clear if save/load don't work)
- **US4 (P3)**: Depends on US1/US2 (can't show recovery if no recovery exists)

### Within Each Phase

- Serializers (T004, T005) can run in parallel
- Storage backend (T006) independent
- SessionStore methods (T008-T012) depend on serializers and storage
- App integration tasks depend on SessionStore being complete

### Parallel Opportunities

```bash
# Phase 2 - Run serializers in parallel:
Task: T004 "Implement DataFrame serializer"
Task: T005 "Implement Graph serializer"

# Phase 6 - Run polish tasks in parallel:
Task: T035 "Handle localStorage quota exceeded"
Task: T036 "Handle corrupted session data"
Task: T037 "Handle version mismatch"
Task: T038 "Add compression for large data"
```

---

## Implementation Strategy

### MVP First (US1 + US2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (serializers, storage, SessionStore)
3. Complete Phase 3: User Stories 1 & 2 (file + form persistence)
4. **STOP and VALIDATE**: Test refresh persistence works end-to-end
5. Deploy/demo if ready - users can already save time on refresh!

### Incremental Delivery

1. **MVP**: Setup + Foundational + US1&2 → Refresh persistence works
2. **+US3**: Add "Start Fresh" button → Users can reset intentionally
3. **+US4**: Add recovery banner → Polished UX with clear feedback
4. **+Polish**: Error handling, compression → Production-ready robustness

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 and US2 combined because they share the same infrastructure
- Manual browser testing required (refresh, close/reopen tab)
- localStorage limit ~10MB per domain - compression helps
- Commit after each task or logical group

# Task List — Feature & Enhancement Ideas

A running list of work for BPSA (Beyond Python SmolAgents). Each entry names
the problem, the change, and the code it touches, so the next agent can act
on it without re-deriving the context. Mark an entry `[x]` when it lands and
append a short "LANDED" note with what was verified.

## Conventions for every task below

- **Never break the application on a missing field.** Every task that reads
  optional data from a provider response (`usage.prompt_tokens_details`,
  `provider`, `cost`, `context_length`, ...) must treat the field as absent
  by default. A provider that omits it, returns `None`, or returns an
  unexpected type must produce the same behaviour as today: no exception,
  no changed output except the new figure being left out. Test each reader
  with a response object that lacks the field.
- **One env var per knob, prefixed `BPSA_`.** Document each new variable in
  `docs/CLI.md` (Environment Variables table) and in the `--help` text in
  `src/smolagents/bp_cli.py`.
- **`OpenAIServerModel` is the primary target.** OpenRouter, DeepSeek, Poe and
  OpenAI all speak this protocol. Other model classes must keep working
  unchanged; a feature that cannot be implemented for them is simply skipped
  for them.
- **Follow `../neural-api/docs/CODING-AND-COMMUNICATION-GUIDE.md`**: look for
  existing code before adding any, name the actor in every report, and do not
  write code unless asked.

## Baseline (existing test suite)

- Code commit: `a8fb19f` (branch `a1`), date 2026-09-14. Only `tasklist.md` has changed since.
- Command (run under the 3 GB cap the coding guide requires):
  `cd /home/bpsa/app/bpsa && ( ulimit -v 3145728; python -m pytest ./tests/ -q -p no:cacheprovider --continue-on-collection-errors )`
  Without `--continue-on-collection-errors` pytest stops at collection ("Interrupted: 4 errors during collection") and runs nothing.
- Environment: `/home/bpsa/x` venv, Python 3.12.3, pytest 9.1.1, pytest-timeout 2.4.0, pytest-datadir 1.8.0. The `[test]` extra (pandas, ipython, mlx, ...) is deliberately not installed.
- Counts: **684 passed, 101 failed, 39 skipped, 4 errors, 0 xfailed**; 823 collected; wall time 223.95 s (3:43). Three tests take 60 s each waiting on a timeout (`test_generation_errors_are_raised`, `test_code_agent_metrics_generation_error`, `test_no_token_usage[CodeAgent]`).

Coding agents must run the same command, compare their counts against these numbers, and leave every failure listed below alone: fixing a pre-existing failure is a separate task, not a side effect.

### Collection errors (4) - environment-caused, missing optional module

- [ ] `tests/test_local_python_executor.py` - ModuleNotFoundError: `pandas`
- [ ] `tests/test_remote_executors.py` - ModuleNotFoundError: `docker`
- [ ] `tests/test_utils.py` - ModuleNotFoundError: `IPython`
- [ ] `tests/test_vision_web_browser.py` - ModuleNotFoundError: `helium (imported by `smolagents/vision_web_browser.py:5`)`

### Failures caused by a missing optional dependency or by the environment (67)

- [ ] `tests/test_agents.py::TestAgent::test_transformers_toolcalling_agent` - environment: needs a huggingface.co model download (fails on this machine)
- [ ] `tests/test_agents.py::TestCodeAgent::test_from_folder[v1.10]` - missing optional dependency `pandas`
- [ ] `tests/test_agents.py::TestCodeAgent::test_from_folder[v1.20]` - missing optional dependency `pandas`
- [ ] `tests/test_agents.py::TestCodeAgent::test_from_dict` - missing optional dependency `pandas`
- [ ] `tests/test_agents.py::TestMultiAgents::test_multiagents_save` - missing optional dependency `pandas`
- [ ] `tests/test_cli.py::test_load_model_litellm_model` - missing optional dependency `litellm`
- [ ] `tests/test_cli.py::test_vision_web_browser_main` - missing optional dependency `helium` (`smolagents.vision_web_browser` import fails)
- [ ] `tests/test_default_tools.py::DefaultToolTests::test_visit_webpage` - environment: TLS certificate verification to huggingface.co fails
- [ ] `tests/test_default_tools.py::TestSpeechToTextTool::test_new_instance` - missing optional dependency `transformers`
- [ ] `tests/test_default_tools.py::TestSpeechToTextTool::test_initialization` - missing optional dependency `transformers`
- [ ] `tests/test_default_tools.py::test_wikipedia_search[en-summary-HTML-Python_(programming_language)]` - missing optional dependency `wikipediaapi`
- [ ] `tests/test_default_tools.py::test_wikipedia_search[en-text-WIKI-Python_(programming_language)]` - missing optional dependency `wikipediaapi`
- [ ] `tests/test_default_tools.py::test_wikipedia_search[es-summary-HTML-Python_(lenguaje_de_programaci\xf3n)]` - missing optional dependency `wikipediaapi`
- [ ] `tests/test_default_tools.py::test_wikipedia_search[es-text-WIKI-Python_(lenguaje_de_programaci\xf3n)]` - missing optional dependency `wikipediaapi`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_custom_types` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_default_types` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_default_types_disallowed` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_invalid_type` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_none` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_special_chars` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::GradioUITester::test_upload_file_success` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_memory_step` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_stream_delta` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_multiple_deltas` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_parameters[simple task-None-False-None]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_parameters[task with images-task_images1-False-None]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_parameters[task with reset-None-True-None]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_parameters[task with args-None-False-additional_args3]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestStreamToGradio::test_stream_to_gradio_parameters[complex task-task_images4-True-additional_args4]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_basic` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_with_tool_calls` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_tool_call_formats[python_interpreter-print('Hello')-```python\nprint('Hello')\n```]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_tool_call_formats[regular_tool-args1-{'key': 'value'}]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_tool_call_formats[string_args_tool-simple string-simple string]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_with_error` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_action_step_with_images` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_planning_step[False-4-token_usage0]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_planning_step[True-2-None]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_final_answer_step[AgentText-This is a text answer-**Final answer:**\nThis is a text answer\n]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_final_answer_step[<lambda>-Plain string-**Final answer:** Plain string]` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_final_answer_step_image` - missing optional dependency `gradio`
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_final_answer_step_audio` - missing optional dependency `soundfile` (audio extra)
- [ ] `tests/test_gradio_ui.py::TestPullMessagesFromStep::test_unsupported_step_type` - missing optional dependency `gradio`
- [ ] `tests/test_import.py::test_import_smolagents_without_extras` - environment: `uv` binary not on PATH
- [ ] `tests/test_models.py::TestModel::test_transformers_message_no_tool` - environment: needs a huggingface.co model download (fails on this machine)
- [ ] `tests/test_models.py::TestModel::test_transformers_message_vl_no_tool` - environment: needs a huggingface.co model download (fails on this machine)
- [ ] `tests/test_models.py::TestLiteLLMModel::test_call_different_providers_without_key[groq/llama-3.3-70b]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMModel::test_call_different_providers_without_key[cerebras/llama-3.3-70b]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMModel::test_call_different_providers_without_key[mistral/mistral-tiny]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMModel::test_retry_on_rate_limit_error` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMModel::test_passing_flatten_messages` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMRouterModel::test_flatten_messages_as_text[llama-3.3-70b-False]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMRouterModel::test_flatten_messages_as_text[llama-3.3-70b-True]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMRouterModel::test_flatten_messages_as_text[mistral-tiny-True]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestLiteLLMRouterModel::test_create_client` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::TestAmazonBedrockModel::test_client_for_bedrock` - missing optional dependency `boto3`
- [ ] `tests/test_models.py::test_flatten_messages_as_text_for_all_models[LiteLLMModel-model_kwargs2-None-False]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::test_flatten_messages_as_text_for_all_models[LiteLLMModel-model_kwargs3-None-True]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::test_flatten_messages_as_text_for_all_models[LiteLLMModel-model_kwargs4-None-True]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::test_flatten_messages_as_text_for_all_models[LiteLLMModel-model_kwargs5-None-True]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::test_flatten_messages_as_text_for_all_models[MLXModel-model_kwargs6-patching6-True]` - missing optional dependency `mlx_lm`
- [ ] `tests/test_models.py::test_tool_calls_json_serialization[LiteLLMModel-gpt-4o-mini]` - missing optional dependency `litellm`
- [ ] `tests/test_models.py::test_tool_calls_json_serialization[OpenAIModel-gpt-4o-mini]` - environment: `OPENAI_API_KEY` not set
- [ ] `tests/test_monitoring.py::MonitoringTester::test_streaming_agent_image_output` - missing optional dependency `gradio`
- [ ] `tests/test_monitoring.py::MonitoringTester::test_streaming_agent_text_output` - missing optional dependency `gradio`
- [ ] `tests/test_monitoring.py::MonitoringTester::test_streaming_with_agent_error` - missing optional dependency `gradio`
- [ ] `tests/test_tools.py::test_launch_gradio_demo_does_not_raise[boolean_default_tool_class]` - missing optional dependency `gradio`

### Failures not explained by the environment (34)

- [ ] `tests/test_agents.py::TestAgent::test_fake_code_agent` - AssertionError: assert False
- [ ] `tests/test_agents.py::TestAgent::test_reset_conversations` - AssertionError: assert '7.2904' == 7.2904
- [ ] `tests/test_agents.py::TestAgent::test_module_imports_get_baked_in_system_prompt` - AssertionError: assert 'collections' in 'You are a super-intelligent assistant who can solve any task.\nYou are running inside Beyond Python Smolag...
- [ ] `tests/test_agents.py::TestAgent::test_init_agent_with_different_toolsets` - AssertionError: assert 6 == 1
- [ ] `tests/test_agents.py::TestAgent::test_function_persistence_across_steps` - ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
- [ ] `tests/test_agents.py::TestAgent::test_agent_description_gets_correctly_inserted_in_system_prompt` - Exception: Managed agents are not yet supported with remote code execution.
- [ ] `tests/test_agents.py::TestAgent::test_final_answer_checks` - AssertionError: assert 'Max steps reached (20/20) - should I continue?' == 7.2904
- [ ] `tests/test_agents.py::TestAgent::test_final_answer_checks_with_agent_access` - AssertionError: assert '7.2904' == 7.2904
- [ ] `tests/test_agents.py::TestAgent::test_planning_step_with_injected_memory` - AssertionError: First planning step should have 4 messages: system-plan-pre-update + memory + task + user-plan-post-update
- [ ] `tests/test_agents.py::TestRunResult::test_no_token_usage[CodeAgent]` - AttributeError: 'FakeLLMModel' object has no attribute 'postpend_string'
- [ ] `tests/test_agents.py::TestMultiStepAgent::test_step_number` - TypeError: unsupported format string passed to MagicMock.__format__
- [ ] `tests/test_agents.py::TestMultiStepAgent::test_interrupt` - TypeError: unsupported format string passed to MagicMock.__format__
- [ ] `tests/test_agents.py::TestMultiStepAgent::test_from_dict` - AssertionError: assert ['final_answe...ionstep', ...] == ['final_answe...ool_function']
- [ ] `tests/test_agents.py::TestMultiStepAgent::test_multiagent_to_dict_from_dict_roundtrip` - ValueError: Tool validation failed for MoveActionStepToMemory:
- [ ] `tests/test_agents.py::TestCodeAgent::test_code_agent_image_output` - AssertionError: assert False
- [ ] `tests/test_agents.py::TestCodeAgent::test_errors_logging` - assert 'secret\\\\' in "'[System prompt: 16,308 chars | Instructions: 12,173 chars | Tool descriptions: \\n4,135 chars]\\n━━━━━━━━━━━━━━━━━━━...\\n...
- [ ] `tests/test_agents.py::TestCodeAgent::test_missing_import_triggers_advice_in_error_log` - AssertionError: assert '`additional_authorized_imports`' in '[System prompt: 16,308 chars | Instructions: 12,173 chars | Tool descriptions: 4,135 c...
- [ ] `tests/test_agents.py::TestCodeAgent::test_errors_show_offending_line_and_error` - assert "Code execution failed at line 'error_function()'" in 'Error executing code: error\nTraceback (most recent call last):\n File "/home/bpsa/x/...
- [ ] `tests/test_agents.py::TestCodeAgent::test_error_saves_previous_print_outputs` - AssertionError: assert 'Flag!' in 'None'
- [ ] `tests/test_agents.py::TestCodeAgent::test_end_code_appending` - assert False
- [ ] `tests/test_agents.py::TestCodeAgent::test_local_python_executor_with_custom_functions` - TypeError: unsupported format string passed to MagicMock.__format__
- [ ] `tests/test_agents.py::TestCodeAgent::test_from_folder[v1.9]` - AssertionError: assert 'exec' == 'local'
- [ ] `tests/test_agents.py::TestCodeAgent::test_custom_final_answer_with_custom_inputs` - TypeError: unsupported format string passed to MagicMock.__format__
- [ ] `tests/test_agents.py::TestCodeAgent::test_use_structured_outputs_internally` - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
- [ ] `tests/test_agents.py::TestMultiAgents::test_multiagents` - Exception: Managed agents are not yet supported with remote code execution.
- [ ] `tests/test_compression.py::TestCompressionConfig::test_default_values` - assert 40 == 5
- [ ] `tests/test_compression.py::TestCompressionConfig::test_keep_compressed_steps_default` - assert 25 == 22
- [ ] `tests/test_compression.py::TestCompressedHistoryStep::test_to_messages` - AssertionError: assert <MessageRole....: 'assistant'> == <MessageRole.USER: 'user'>
- [ ] `tests/test_compression.py::TestCreateCompressionPrompt::test_creates_prompt_for_action_steps` - AssertionError: assert 'Summarize' in 'Hello super-intelligence!\nThis task is involved in your context compression.\nPlease summarize the followin...
- [ ] `tests/test_final_answer.py::TestFinalAnswerTool::test_agent_type_output` - AssertionError: assert False
- [ ] `tests/test_memory.py::test_system_prompt_step_to_messages` - AssertionError: assert <MessageRole.USER: 'user'> == <MessageRole.SYSTEM: 'system'>
- [ ] `tests/test_models.py::TestModel::test_prepare_completion_kwargs_parameter_precedence` - AssertionError: assert 'max_tokens' not in {'messages': [{'role': <MessageRole.USER: 'user'>, 'content': [{'type': 'text', 'text': 'Hello'}]}], 'ma...
- [ ] `tests/test_models.py::test_get_clean_message_list_image_encoding[False-expected_clean_message0]` - AttributeError: 'bytes' object has no attribute 'save'
- [ ] `tests/test_models.py::test_get_clean_message_list_image_encoding[True-expected_clean_message1]` - AttributeError: 'bytes' object has no attribute 'save'

## Provider visibility (turn summary and `/show-stats`)

These share one blind spot: `TokenUsage` in `src/smolagents/monitoring.py`
holds only `input_tokens` and `output_tokens`, so the REPL cannot show whether
`BPSA_HAS_SESSION_ID`, `BPSA_PROVIDER_ORDER` and `BPSA_SYSTEM_PROMPT_FIRST`
are paying off. `OpenAIModel.generate` and `OpenAIModel.generate_stream` in
`src/smolagents/models.py` already hold the full response object and drop
these fields when they build `TokenUsage`.

- [x] **Cache hit rate per turn.** OpenRouter returns
      `usage.prompt_tokens_details.cached_tokens`, DeepSeek returns
      `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens`,
      OpenAI returns `usage.prompt_tokens_details.cached_tokens`. Add a
      `cached_input_tokens: int = 0` field to `TokenUsage`, fill it in
      `OpenAIModel.generate` / `generate_stream` from whichever field is
      present (0 when none is), and carry it through
      `agglomerate_stream_deltas` and `Monitor.get_total_token_counts`.
      `print_turn_summary` in `bp_cli.py` then prints `Cache: 87%`
      (cached / input) when the turn had any cached tokens. This lets the
      user see immediately whether `BPSA_HAS_SESSION_ID` is working.
      Must not break when `usage` is `None` or lacks the detail object.
      LANDED (commit 2cf08b5): `TokenUsage.cached_input_tokens`,
      `extract_cached_input_tokens` in `models.py` (used by
      `OpenAIModel.generate` / `generate_stream`), summed in
      `agglomerate_stream_deltas` and `Monitor`, persisted by `bp_session.py`
      (old files load with 0), shown by `print_turn_summary` (`Cache: NN%`)
      and `print_stats` ("Total cached input tokens"). Tests:
      `tests/test_bp_token_cache.py` (12 new). Full suite: 697 passed,
      100 failed, 39 skipped, 4 errors vs baseline 684/101/39/4; the failed
      set is the baseline set minus `test_visit_webpage` (network, passed
      this time). `test_memory.py::test_action_step_dict` expects the new
      key in `TokenUsage.dict()`. Live check on OpenRouter with
      `~deepseek/deepseek-flash-latest`: the response carries
      `prompt_tokens_details.cached_tokens` and the reader picks it up, but
      OpenRouter routed consecutive calls to different providers (Relace,
      DeepInfra) even with `BPSA_HAS_SESSION_ID=1` and
      `BPSA_PROVIDER_ORDER=deepseek`, so every call reported 0 cached tokens
      and the turn summary correctly showed no `Cache:` figure.
- [x] **Show which provider served the request.** OpenRouter puts a
      `provider` string in every response body (`response.provider` on the
      OpenAI SDK object, reachable via `getattr` or `model_extra`). Store it
      on the `ChatMessage` (or on `TokenUsage` as `provider: str | None`)
      and print `via DeepInfra` in the turn summary. Verifies
      `BPSA_PROVIDER_ORDER` and sticky routing without guessing. When the
      field is absent (OpenAI, DeepSeek), print nothing.
      LANDED (commit 19362bb): `TokenUsage.provider` (None by default),
      `extract_provider_name` in `models.py` (reads `provider` from the
      object, dict or `model_extra`; None for missing/None/non-string), used
      by `OpenAIModel.generate` / `generate_stream` (streaming keeps the
      last chunk-level provider seen and attaches it to the usage delta);
      `agglomerate_stream_deltas` keeps the last non-None provider;
      `CodeAgent._step_stream` in `agents.py` now keeps the aggregated
      `TokenUsage` (it rebuilt one from bare counts, which dropped the
      field); `Monitor.last_provider` holds the most recent step's provider
      (reset with the counters) and `bp_session.py` persists it plus the
      per-step field (old files load with None); `print_turn_summary`
      appends `via <Provider>` from the monitor and `print_stats` shows
      "Last provider". Docs: `docs/CLI.md` (paragraph after the Cache one).
      Tests: `tests/test_bp_provider_name.py` (16 new). Full suite: 713
      passed, 100 failed, 39 skipped, 4 errors vs baseline 684/101/39/4;
      the failed set is the baseline set minus `test_visit_webpage`
      (network, passed this time). Live check on OpenRouter with
      `~deepseek/deepseek-flash-latest`, `BPSA_HAS_SESSION_ID=1`: turn 1
      printed `... | via DeepInfra | Auto-approve: off`, turn 2 printed
      `... | via Parasail | Auto-approve: on`, `/show-stats` printed
      "Last provider  Parasail"; OpenRouter again routed consecutive calls
      to different providers, and both reported 0 cached tokens.
- [ ] **Cost per turn and per session.** OpenRouter reports exact cost when
      the request body contains `usage: {"include": true}` — one more entry
      in the `extra_body` that `build_model` in `bp_cli.py` already
      assembles; the response then carries `usage.cost`. For DeepSeek and
      OpenAI add two optional env vars, `BPSA_PRICE_INPUT_PER_M` and
      `BPSA_PRICE_OUTPUT_PER_M` (USD per million tokens), and estimate cost
      from `TokenUsage`; count cached input tokens at the input price unless
      a `BPSA_PRICE_CACHED_INPUT_PER_M` is given. Add `cost_usd: float`
      to `TokenUsage`, accumulate it in `session_stats` next to
      `total_input_tokens`, show it in `print_turn_summary` and
      `print_stats`, and persist it through `/session-save` /
      `/session-load`. Money is what users actually budget by. When
      neither the response cost nor the price env vars are available, show
      no cost line.

## Configuration ergonomics

- [ ] **`/model <id>` to switch the main model mid-session.**
      `build_model(override_model_id=...)` in `bp_cli.py` already accepts an
      override id, and `cmd_compression_model` already does this for the
      compression model. Rebuild the main model with the new id, keep
      `agent.memory` intact, assign `agent.model`, and update the banner /
      `session_stats` model name. Lets a user draft with a cheap model and
      finish with a strong one. `/model` with no argument prints the current
      model id. Add the command to the completer list, `/help` table and
      `docs/CLI.md`.
- [ ] **`/show-config`.** Print the effective settings in one table: model
      class, endpoint, model id, masked API key (first 4 and last 4
      characters), provider order, current OpenRouter session id
      (`current_session_id()`), system-prompt position
      (`BPSA_SYSTEM_PROMPT_FIRST`), executor, max steps, planning interval,
      compression thresholds (`agent.compression_config`), and which
      optional tool sets are enabled (browser, gui, image, tmux, mcp). Right
      now the only way to check any of this is to inspect the environment by
      hand. Note that `docs/CLI.md` promises a `~/.bpsa.yaml` config layer
      that `bp_cli.py` never reads; `/show-config` should state the actual
      source of each value (env, `.env`, default).
- [ ] **Startup connectivity check.** After `build_model` and before the
      banner, send one tiny request (a single short user message with
      `max_tokens` around 8) and fail with a clear message on a bad key,
      wrong endpoint, or unknown model id. This matches the "fail fast" rule
      in `docs/CLI.md` (Startup Behavior). Today the first failure appears
      only after the user has typed a full prompt. Skip the check for local
      model classes (`TransformersModel`, `MLXModel`, `VLLMModel`) and allow
      `BPSA_SKIP_CONNECTIVITY_CHECK=1` to bypass it. Report the round-trip
      time in the banner so the user also sees endpoint latency.

## Context awareness and budgets

- [ ] **Know the model's context length.** OpenRouter's `GET /models`
      endpoint returns `context_length` per model id; OpenAI-compatible
      endpoints that do not serve it fall back to a `BPSA_CONTEXT_LENGTH`
      env var. With that number `print_turn_summary` can show `Context: 61%`
      (last-turn `input_tokens` over the window) instead of a raw character
      count, and `BPSA_COMPRESSION_TOKEN_THRESHOLD` (currently `0` = off)
      can default to a fraction of the real window. Fetch once at startup,
      never on the hot path; on any failure keep the current character
      display.
- [ ] **Session budget.** `BPSA_MAX_SESSION_TOKENS` and
      `BPSA_MAX_SESSION_COST` (USD, requires the cost task above). The REPL
      warns once at 80% and stops accepting new agent turns at 100%
      (slash commands still work so the user can `/session-save`).
      `ad-infinitum` (`bp_ad_infinitum.py`) stops the cycle loop at 100% and
      exits with a non-zero code. Protects unattended runs from an open-ended
      bill. `0` or unset means no limit.

## Interaction while the agent is working

- [ ] **Steer while it runs.** A background thread reads lines while the
      agent works and puts them in a queue. At the next step boundary the
      agent drains the queue into memory, and the model sees
      "user update: also run the tests" at its next call. Non-blocking for
      the user, no step lost. This is the interaction feature the user
      misses most.
      Mechanics: today the REPL thread sits inside `agent.run()` and
      `prompt_toolkit` is not running, so typed text goes nowhere and the
      only escape is Ctrl+C (a hard abort that loses the in-flight step).
      `_run_stream` in `src/smolagents/agents.py` already checks
      `interrupt_switch` at the top of every step; drain the queue there.
      `write_memory_to_messages` rebuilds the context before every model
      call, so anything appended to memory between steps is seen on the
      next call with no other change.
      Decisions: (a) no new step class. When a message is queued, the
      agent drains it at the step boundary and appends it to the messages
      sent to the model as a normal user-role message, placed at the end,
      after the run output of the step that just finished. To keep it in
      context on later calls, store the text on the finished `ActionStep`
      (a `user_message` field next to `observations`) and have
      `ActionStep.to_messages` in `src/smolagents/memory.py` emit it as one
      `MessageRole.USER` message right after the observation message; the
      step's `dict()` then carries it through compression and
      `/session-save` with no other change;
      (b) use `prompt_toolkit`'s `patch_stdout` so the input line stays
      pinned while step output scrolls above it; (c) one listener thread
      owns stdin and hands lines to whichever consumer is waiting (the
      steering queue, `interactive_approval_callback`, a future
      `ask_user` tool), otherwise two readers lose keystrokes;
      (d) add one sentence to the system prompt saying user updates may
      arrive between steps and take priority over the original task;
      (e) disable the listener when stdin is not a tty; (f) let
      `ad-infinitum` feed the same queue from a file such as
      `tasks/_inbox.md`, which also makes the mechanism testable without a
      terminal.

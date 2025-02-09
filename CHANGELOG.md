# Changelog

## Unreleased

### Breaking Changes
...
### New Features
...
### Fixes
...
### Deprecations
...

## 1.2.0

We did a major revamp of the `ArgillaEvaluator` to separate an `AsyncEvaluator` from the normal evaluation scenario.
This comes with easier to understand interfaces, more information in the `EvaluationOverview` and a simplified aggregation step for Argilla that is no longer dependent on specific Argilla types.
Check the how-to for detailed information [here](./src/documentation/how_tos/how_to_human_evaluation_via_argilla.ipynb)

### Breaking Changes

- rename: `AggregatedInstructComparison` to `AggregatedComparison`
- rename `InstructComparisonArgillaAggregationLogic` to `ComparisonAggregationLogic`
- remove: `ArgillaAggregator` - the regular aggregator now does the job
- remove: `ArgillaEvaluationRepository` - `ArgillaEvaluator` now uses `AsyncRepository` which extend existing `EvaluationRepository` for the human-feedback use-case
- `ArgillaEvaluationLogic` now uses `to_record` and `from_record` instead of `do_evaluate`. The signature of the `to_record` stays the same. The `Field` and `Question` are now defined in the logic instead of passed to the `ArgillaRepository`
- `ArgillaEvaluator` now takes the `ArgillaClient` as well as the `workspace_id`. It inherits from the abstract `AsyncEvaluator` and no longer has `evalaute_runs` and `evaluate`. Instead it has `submit` and `retrieve`.
- `EvaluationOverview` gets attributes `end_date`, `successful_evaluation_count` and `failed_evaluation_count`
  - rename: `start` is now called `start_date` and no longer optional
- we refactored the internals of `Evaluator`. This is only relevant if you subclass from it. Most of the typing and data handling is moved to `EvaluatorBase`

### New Features
- Add `ComparisonEvaluation` for the elo evaluation to abstract from the Argilla record
- Add `AsyncEvaluator` for human-feedback evaluation. `ArgillaEvaluator` inherits from this
  - `.submit` pushes all evaluations to Argilla to label them
  - Add `PartialEvaluationOverview` to store the submission details.
  - `.retrieve` then collects all labelled records from Argilla and stores them in an `AsyncRepository`.
  - Add `AsyncEvaluationRepository` to store and retrieve `PartialEvaluationOverview`. Also added `AsyncFileEvaluationRepository` and `AsyncInMemoryEvaluationRepository`
- Add `EvaluatorBase` and `EvaluationLogicBase` for base classes for both async and synchronous evaluation.

### Fixes
 - Improve description of using artifactory tokens for installation of IL
 - Change `confusion_matrix` in `SingleLabelClassifyAggregationLogic` such that it can be persisted in a file repository

## 1.1.0

### New Features
 - `AlephAlphaModel` now supports a `context_size`-property
 - Add new `IncrementalEvaluator` for easier addition of runs to existing evaluations without repeated evaluation.
   - Add `IncrementalEvaluationLogic` for use in `IncrementalEvaluator`

## 1.0.0

Initial stable release

With the release of version 1.0.0 there have been introduced some new features but also some breaking changes you should be aware of.
Apart from these changes, we also had to reset our commit history, so please be aware of this fact.

### Breaking Changes
-  The TraceViewer has been exported to its own repository and can be accessed via the artifactory [here]( https://alephalpha.jfrog.io.)
-  `HuggingFaceDatasetRepository` now has a parameter caching, which caches  examples of a dataset once loaded.
  - `True` as default value
  - set to `False` for **non-breaking**-change


### New Features
#### Llama2 and LLama3 model support
-  Introduction of `LLama2InstructModel` allows support of the LLama2-models:
  - `llama-2-7b-chat`
  - `llama-2-13b-chat`
  - `llama-2-70b-chat`
-  Introduction of `LLama3InstructModel` allows support of the LLama2-models:
  - `llama-3-8b-instruct`
  - `llama-3-70b-instruct`
#### DocumentIndexClient
`DocumentIndexClient` has been enhanced with the following set of features:
-  `create_index`
- feature `index_configuration`
-  `assign_index_to_collection`
-  `delete_index_from_collection`
-  `list_assigned_index_names`

#### Miscellaneous
-  `ExpandChunks`-task now caches chunked documents by ID
-  `DocumentIndexRetriever` now supports `index_name`
-  `Runner.run_dataset` now has a configurable number of workers via `max_workers` and defaults to the previous value, which is 10.
-  In case a `BusyError` is raised during a `complete` the `LimitedConcurrencyClient` will retry until `max_retry_time` is reached.

### Fixes
-  `HuggingFaceRepository` no longer is a dataset repository. This also means that `HuggingFaceAggregationRepository` no longer is a dataset repository.
-  The input parameter of the `DocumentIndex.search()`-function now has been renamed from `index` to `index_name`

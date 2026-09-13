"""Tasks 3-5 - indexing, retrieval, grounded generation, precision/recall."""


from __future__ import annotations


import pytest


from app.config import FALLBACK_ANSWER, STRATEGY_FIXED, STRATEGY_SENTENCE, Settings
from rag.calibration import (
    IN_SCOPE_CALIBRATION_QUERIES,
    OUT_OF_SCOPE_CALIBRATION_QUERIES,
    CalibrationOverlapError,
    CalibrationResult,
    QueryMeasurement,
    measure,
)
from rag.chunking import KbDocument
from rag.embeddings import DeterministicHashEmbedder, cosine_similarity, l2_normalise
from rag.evaluation import (
    DEMO_QUERIES,
    IN_SCOPE_DEMO_QUERIES,
    OUT_OF_SCOPE_DEMO_QUERIES,
    QueryScore,
    recommend_strategy,
    score_collection,
)
from rag.grounded_generation import (
    CalibrationRequiredError,
    GroundedGenerator,
    ThresholdSource,
    resolve_threshold,
)
from rag.indexer import build_indexes
from rag.retriever import RetrievalError, Retriever
from rag.textutils import overlap_ratio, select_relevant_sentences
from tests.fakes import FakeChromaClient, FakeEmbedder




class TestIndexing:
    def test_each_strategy_gets_its_own_collection(
        self, built_index: FakeChromaClient, test_settings: Settings
    ) -> None:
        assert set(built_index.collections) == set(test_settings.all_collection_names)
        for name in test_settings.all_collection_names:
            assert built_index.collections[name].count() > 0


    def test_a_collection_holds_only_its_own_strategy(
        self, built_index: FakeChromaClient, test_settings: Settings
    ) -> None:
        expected = test_settings.strategy_for_collection
        for name, collection in built_index.collections.items():
            result = collection.query(
                query_embeddings=[[0.0] * 256], n_results=collection.count()
            )
            for metadata in result["metadatas"][0]:
                assert metadata["strategy"] == expected[name]


    def test_rebuild_is_idempotent(
        self,
        test_settings: Settings,
        built_index: FakeChromaClient,
        embedder: FakeEmbedder,
        documents: list[KbDocument],
        index_state_file,
    ) -> None:
        before = {
            name: collection.count() for name, collection in built_index.collections.items()
        }
        build_indexes(
            test_settings,
            client=built_index,
            embedder=embedder,
            documents=documents,
            reset=True,
        )
        after = {
            name: collection.count() for name, collection in built_index.collections.items()
        }
        assert before == after


    def test_kb_version_increments_on_each_build(
        self,
        test_settings: Settings,
        chroma_client: FakeChromaClient,
        embedder: FakeEmbedder,
        documents: list[KbDocument],
        index_state_file,
    ) -> None:
        first = build_indexes(
            test_settings, client=chroma_client, embedder=embedder, documents=documents
        )
        second = build_indexes(
            test_settings, client=chroma_client, embedder=embedder, documents=documents
        )
        assert second.kb_version == first.kb_version + 1




class TestRetrieval:
    def test_returns_at_most_top_k(self, retriever: Retriever) -> None:
        result = retriever.search("cancellation window", top_k=2)
        assert len(result.chunks) <= 2


    def test_results_are_ordered_by_distance(self, retriever: Retriever) -> None:
        result = retriever.search("consultation fee for cardiology", top_k=3)
        distances = [chunk.distance for chunk in result.chunks]
        assert distances == sorted(distances)


    def test_similarity_is_one_minus_distance(self, retriever: Retriever) -> None:
        result = retriever.search("prescription refill", top_k=3)
        for chunk in result.chunks:
            assert chunk.similarity == pytest.approx(1.0 - chunk.distance)


    def test_document_ids_are_deduplicated_and_ordered(self, retriever: Retriever) -> None:
        result = retriever.search("cancellation and rescheduling window", top_k=3)
        ids = result.document_ids
        assert len(ids) == len(set(ids))
        assert ids[0] == result.chunks[0].document_id


    def test_empty_query_is_rejected(self, retriever: Retriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.search("   ")


    def test_unknown_collection_is_rejected(self, retriever: Retriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.search("anything", collection_name="not_a_collection")


    def test_empty_collection_is_rejected(
        self, test_settings: Settings, embedder: FakeEmbedder
    ) -> None:
        client = FakeChromaClient()
        empty = Retriever(test_settings, client=client, embedder=embedder)
        with pytest.raises(RetrievalError) as excinfo:
            empty.search("cancellation window")
        assert "build_indexes" in str(excinfo.value)


    def test_query_counter_advances(self, retriever: Retriever) -> None:
        retriever.reset_counters()
        retriever.search("lab test turnaround")
        retriever.search("home visit eligibility")
        assert retriever.query_count == 2




class TestGroundedGeneration:
    def test_in_scope_queries_are_grounded(self, generator: GroundedGenerator) -> None:
        for demo in IN_SCOPE_DEMO_QUERIES:
            answer = generator.generate(demo.query)
            assert answer.grounded, demo.query_id
            assert answer.sources, demo.query_id
            assert answer.answer != FALLBACK_ANSWER


    def test_out_of_scope_query_falls_back(self, generator: GroundedGenerator) -> None:
        answer = generator.generate(OUT_OF_SCOPE_DEMO_QUERIES[0].query)
        assert not answer.grounded
        assert answer.answer == FALLBACK_ANSWER
        assert answer.sources == ()


    def test_answer_text_comes_only_from_retrieved_context(
        self, generator: GroundedGenerator
    ) -> None:
        answer = generator.generate(IN_SCOPE_DEMO_QUERIES[0].query)
        # Every quoted sentence must be present verbatim in the context. The
        # first sentence carries the frame, so it is checked by overlap instead.
        from rag.textutils import split_sentences


        sentences = split_sentences(answer.answer)
        for sentence in sentences[1:]:
            assert sentence in answer.context_text


    def test_a_threshold_above_every_similarity_forces_the_fallback(
        self, retriever: Retriever, test_settings: Settings
    ) -> None:
        impossible = GroundedGenerator(
            retriever=retriever,
            settings=test_settings,
            threshold_source=ThresholdSource(1.01, "test", "above any cosine similarity"),
        )
        answer = impossible.generate(IN_SCOPE_DEMO_QUERIES[0].query)
        assert not answer.grounded
        assert answer.answer == FALLBACK_ANSWER


    def test_empty_query_is_rejected(self, generator: GroundedGenerator) -> None:
        with pytest.raises(ValueError):
            generator.generate("  ")


    def test_fallback_counter_advances(self, generator: GroundedGenerator) -> None:
        generator.reset_counters()
        generator.generate(OUT_OF_SCOPE_DEMO_QUERIES[0].query)
        assert generator.stats.fallbacks == 1




class TestThresholdResolution:
    def test_missing_calibration_raises_rather_than_guessing(
        self, test_settings: Settings, tmp_path
    ) -> None:
        with pytest.raises(CalibrationRequiredError) as excinfo:
            resolve_threshold(
                test_settings, calibration_file=tmp_path / "absent.json"
            )
        assert "calibrate_threshold" in str(excinfo.value)


    def test_environment_override_wins(self, test_settings: Settings, tmp_path) -> None:
        import dataclasses


        pinned = dataclasses.replace(test_settings, similarity_threshold_override=0.42)
        source = resolve_threshold(pinned, calibration_file=tmp_path / "absent.json")
        assert source.value == 0.42
        assert source.origin == "environment"


    def test_out_of_range_override_is_rejected(self, test_settings: Settings) -> None:
        import dataclasses


        broken = dataclasses.replace(test_settings, similarity_threshold_override=3.0)
        with pytest.raises(CalibrationRequiredError):
            resolve_threshold(broken)


    def test_unusable_calibration_file_raises(
        self, test_settings: Settings, tmp_path
    ) -> None:
        path = tmp_path / "calibration.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CalibrationRequiredError):
            resolve_threshold(test_settings, calibration_file=path)




class TestCalibration:
    def test_probe_sets_meet_the_brief_minimums(self) -> None:
        assert len(IN_SCOPE_CALIBRATION_QUERIES) >= 3
        assert len(OUT_OF_SCOPE_CALIBRATION_QUERIES) >= 2


    def test_measurement_covers_every_probe(
        self, retriever: Retriever, test_settings: Settings
    ) -> None:
        result = measure(retriever, settings=test_settings)
        assert len(result.in_scope) == len(IN_SCOPE_CALIBRATION_QUERIES)
        assert len(result.out_of_scope) == len(OUT_OF_SCOPE_CALIBRATION_QUERIES)


    def test_threshold_is_the_midpoint_of_the_observed_gap(self) -> None:
        result = CalibrationResult(
            collection_name="c",
            measured_at="now",
            embedder="fake",
            in_scope=(QueryMeasurement("in", 0.60, "doc", True),),
            out_of_scope=(QueryMeasurement("out", 0.20, "doc", False),),
        )
        assert result.separable
        assert result.margin == pytest.approx(0.40)
        assert result.recommended_threshold == pytest.approx(0.40)


    def test_overlapping_clusters_refuse_to_produce_a_threshold(self) -> None:
        result = CalibrationResult(
            collection_name="c",
            measured_at="now",
            embedder="fake",
            in_scope=(QueryMeasurement("in", 0.20, "doc", True),),
            out_of_scope=(QueryMeasurement("out", 0.50, "doc", False),),
        )
        assert not result.separable
        with pytest.raises(CalibrationOverlapError):
            _ = result.recommended_threshold


    def test_rejects_too_few_probes(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError):
            measure(retriever, in_scope=("only one",), out_of_scope=("a", "b"))




class TestPrecisionRecall:
    def test_query_set_matches_the_brief(self) -> None:
        assert len(IN_SCOPE_DEMO_QUERIES) >= 5
        assert len(OUT_OF_SCOPE_DEMO_QUERIES) >= 1
        assert len(DEMO_QUERIES) == len(IN_SCOPE_DEMO_QUERIES) + len(
            OUT_OF_SCOPE_DEMO_QUERIES
        )


    def test_arithmetic_is_correct(self) -> None:
        score = QueryScore(
            query_id="q",
            query="q",
            collection_name="c",
            strategy=STRATEGY_FIXED,
            relevant=frozenset({"a", "b"}),
            retrieved_documents=("a", "c"),
            retrieved_chunk_ids=("x", "y"),
            top_similarity=0.5,
        )
        assert score.true_positives == frozenset({"a"})
        assert score.precision == pytest.approx(0.5)
        assert score.recall == pytest.approx(0.5)
        assert "1/2" in score.precision_arithmetic
        assert "1/2" in score.recall_arithmetic


    def test_zero_retrieved_scores_zero(self) -> None:
        score = QueryScore(
            query_id="q",
            query="q",
            collection_name="c",
            strategy=STRATEGY_SENTENCE,
            relevant=frozenset({"a"}),
            retrieved_documents=(),
            retrieved_chunk_ids=(),
            top_similarity=0.0,
        )
        assert score.precision == 0.0
        assert score.recall == 0.0


    def test_both_collections_are_scored(
        self, retriever: Retriever, test_settings: Settings
    ) -> None:
        scores = {
            name: score_collection(retriever, name, queries=IN_SCOPE_DEMO_QUERIES)
            for name in test_settings.all_collection_names
        }
        assert len(scores) == 2
        for score in scores.values():
            assert len(score.scores) == len(IN_SCOPE_DEMO_QUERIES)
            assert 0.0 <= score.mean_precision <= 1.0
            assert 0.0 <= score.mean_recall <= 1.0


    def test_recommendation_picks_the_higher_f1(
        self, retriever: Retriever, test_settings: Settings
    ) -> None:
        scores = {
            name: score_collection(retriever, name, queries=IN_SCOPE_DEMO_QUERIES)
            for name in test_settings.all_collection_names
        }
        winner, rationale = recommend_strategy(scores)
        assert winner in test_settings.all_collection_names
        best_f1 = max(score.mean_f1 for score in scores.values())
        assert scores[winner].mean_f1 == best_f1
        assert "mean F1" in rationale


    def test_documents_are_deduplicated_before_scoring(
        self, retriever: Retriever, test_settings: Settings
    ) -> None:
        score = score_collection(
            retriever,
            test_settings.fixed_collection_name,
            queries=IN_SCOPE_DEMO_QUERIES,
        )
        for query_score in score.scores:
            docs = list(query_score.retrieved_documents)
            assert len(docs) == len(set(docs))
            assert len(docs) <= len(query_score.retrieved_chunk_ids)




class TestEmbeddingHelpers:
    def test_normalisation_produces_unit_vectors(self) -> None:
        vector = l2_normalise([3.0, 4.0])
        assert vector == pytest.approx([0.6, 0.8])


    def test_zero_vector_survives_normalisation(self) -> None:
        assert l2_normalise([0.0, 0.0]) == [0.0, 0.0]


    def test_cosine_of_identical_vectors_is_one(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


    def test_cosine_rejects_a_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


    def test_hash_embedder_is_stable_across_instances(self) -> None:
        left = DeterministicHashEmbedder(64).encode(["cancellation window"])
        right = DeterministicHashEmbedder(64).encode(["cancellation window"])
        assert left == right


    def test_hash_embedder_rejects_a_bare_string(self) -> None:
        with pytest.raises(TypeError):
            DeterministicHashEmbedder(64).encode("not a sequence")  # type: ignore[arg-type]




class TestTextUtils:
    def test_overlap_ratio_of_a_verbatim_sentence_is_one(self) -> None:
        context = "Cancellation is free up to 4 hours before the appointment."
        assert overlap_ratio(context, context) == pytest.approx(1.0)


    def test_overlap_ratio_of_unrelated_text_is_low(self) -> None:
        assert overlap_ratio(
            "platinum membership unlimited complimentary surgeries",
            "Cancellation is free up to 4 hours before the appointment.",
        ) < 0.3


    def test_function_words_only_counts_as_supported(self) -> None:
        assert overlap_ratio("It is so.", "anything at all") == pytest.approx(1.0)


    def test_selection_returns_verbatim_sentences_in_order(self) -> None:
        context = "Alpha one. Beta two. Gamma three."
        chosen = select_relevant_sentences("gamma", context, 2)
        assert all(sentence in context for sentence in chosen)
        assert chosen == sorted(chosen, key=context.index)


    def test_selection_rejects_a_non_positive_limit(self) -> None:
        with pytest.raises(ValueError):
            select_relevant_sentences("q", "A sentence.", 0)




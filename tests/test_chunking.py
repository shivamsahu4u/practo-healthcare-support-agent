"""Task 3 - chunking strategies, metadata, and knowledge-base loading."""


from __future__ import annotations


import pytest


from app.config import KB_TOPICS, STRATEGY_FIXED, STRATEGY_SENTENCE, Settings
from rag.chunking import (
    KbDocument,
    KnowledgeBaseError,
    chunk_corpus,
    chunk_corpus_all_strategies,
    chunk_document,
    chunk_fixed_size,
    chunk_sentences,
    load_knowledge_base,
    normalise_whitespace,
    parse_document,
    split_sentences,
)




class TestFixedSizeChunking:
    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_fixed_size("A short policy sentence.", 480, 96) == [
            "A short policy sentence."
        ]


    def test_consecutive_chunks_overlap(self) -> None:
        text = " ".join(f"word{index}" for index in range(200))
        chunks = chunk_fixed_size(text, 200, 60)
        assert len(chunks) > 1
        for first, second in zip(chunks, chunks[1:]):
            # The next window starts `overlap` characters before the previous
            # one ended, so the previous window's closing words reappear at the
            # start of the next. The next window may open mid-word, hence the
            # generous head slice.
            tail = set(first.split()[-5:])
            head = set(second.split()[:10])
            assert tail & head, "consecutive fixed-size chunks share no words"


    def test_no_chunk_exceeds_the_window(self) -> None:
        text = " ".join(f"word{index}" for index in range(300))
        for chunk in chunk_fixed_size(text, 180, 40):
            assert len(chunk) <= 180


    def test_words_are_never_split(self) -> None:
        text = " ".join("consultation" for _ in range(60))
        for chunk in chunk_fixed_size(text, 100, 20):
            for token in chunk.split():
                assert token == "consultation"


    def test_empty_input_yields_no_chunks(self) -> None:
        assert chunk_fixed_size("   ", 100, 10) == []


    @pytest.mark.parametrize("size,overlap", [(100, 100), (100, 150), (0, 0), (10, -1)])
    def test_rejects_impossible_parameters(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("some text", size, overlap)


    def test_terminates_on_text_with_no_spaces(self) -> None:
        chunks = chunk_fixed_size("x" * 500, 100, 20)
        assert chunks
        assert "".join(dict.fromkeys(chunks[0])) == "x"




class TestSentenceChunking:
    def test_groups_whole_sentences(self) -> None:
        text = "One. Two. Three. Four. Five."
        assert chunk_sentences(text, 2) == ["One. Two.", "Three. Four.", "Five."]


    def test_one_sentence_per_chunk(self) -> None:
        assert chunk_sentences("One. Two.", 1) == ["One.", "Two."]


    def test_rejects_a_non_positive_group_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_sentences("One. Two.", 0)


    def test_splitting_handles_all_terminators(self) -> None:
        assert split_sentences("A? B! C.") == ["A?", "B!", "C."]




class TestMetadata:
    def test_chunk_metadata_carries_the_parent_document(
        self, documents: list[KbDocument], test_settings: Settings
    ) -> None:
        document = documents[0]
        chunks = chunk_document(document, STRATEGY_FIXED, test_settings)
        assert chunks
        for index, chunk in enumerate(chunks):
            assert chunk.document_id == document.document_id
            assert chunk.source_filename == document.source_filename
            assert chunk.topic_title == document.title
            assert chunk.strategy == STRATEGY_FIXED
            assert chunk.chunk_index == index
            assert chunk.chunk_id.startswith(f"{STRATEGY_FIXED}::{document.document_id}")


    def test_chroma_metadata_is_scalar_only(
        self, documents: list[KbDocument], test_settings: Settings
    ) -> None:
        chunk = chunk_document(documents[0], STRATEGY_SENTENCE, test_settings)[0]
        for value in chunk.metadata().values():
            assert isinstance(value, (str, int, float, bool))


    def test_chunk_ids_are_unique_within_a_strategy(
        self, documents: list[KbDocument], test_settings: Settings
    ) -> None:
        for strategy in (STRATEGY_FIXED, STRATEGY_SENTENCE):
            chunks = chunk_corpus(documents, strategy, test_settings)
            ids = [chunk.chunk_id for chunk in chunks]
            assert len(set(ids)) == len(ids)


    def test_strategies_never_share_a_chunk_id(
        self, documents: list[KbDocument], test_settings: Settings
    ) -> None:
        chunked = chunk_corpus_all_strategies(documents, test_settings)
        fixed = {chunk.chunk_id for chunk in chunked[STRATEGY_FIXED]}
        sentence = {chunk.chunk_id for chunk in chunked[STRATEGY_SENTENCE]}
        assert not fixed & sentence


    def test_rejects_an_unknown_strategy(
        self, documents: list[KbDocument], test_settings: Settings
    ) -> None:
        with pytest.raises(ValueError):
            chunk_document(documents[0], "no_such_strategy", test_settings)




class TestKnowledgeBase:
    def test_all_required_topics_are_present(self, documents: list[KbDocument]) -> None:
        assert len(documents) >= 12
        present = {document.document_id for document in documents}
        for topic in KB_TOPICS:
            assert topic.slug in present, topic.slug


    def test_every_required_document_has_two_to_five_sentences(
        self, documents: list[KbDocument]
    ) -> None:
        for document in documents:
            if not document.is_required_topic:
                continue
            count = len(split_sentences(document.body))
            assert 2 <= count <= 5, f"{document.document_id} has {count} sentences"


    def test_documents_are_sorted_for_determinism(
        self, documents: list[KbDocument]
    ) -> None:
        assert [d.document_id for d in documents] == sorted(
            d.document_id for d in documents
        )


    def test_title_is_stripped_from_the_body(self, documents: list[KbDocument]) -> None:
        for document in documents:
            assert not document.body.startswith("#")
            assert document.title


    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(KnowledgeBaseError):
            load_knowledge_base(tmp_path / "does-not-exist")


    def test_empty_directory_raises(self, tmp_path) -> None:
        with pytest.raises(KnowledgeBaseError):
            load_knowledge_base(tmp_path)


    def test_document_without_a_heading_raises(self, tmp_path) -> None:
        path = tmp_path / "broken.md"
        path.write_text("No heading here, just prose.", encoding="utf-8")
        with pytest.raises(KnowledgeBaseError):
            parse_document(path)


    def test_document_without_a_body_raises(self, tmp_path) -> None:
        path = tmp_path / "empty.md"
        path.write_text("# Only A Heading\n", encoding="utf-8")
        with pytest.raises(KnowledgeBaseError):
            parse_document(path)


    def test_incomplete_knowledge_base_raises(self, tmp_path) -> None:
        (tmp_path / "appointment_booking.md").write_text(
            "# Appointment Booking Policy\n\nOne sentence. Two sentences.\n",
            encoding="utf-8",
        )
        with pytest.raises(KnowledgeBaseError) as excinfo:
            load_knowledge_base(tmp_path)
        assert "missing required topic" in str(excinfo.value)




def test_normalise_whitespace_collapses_runs() -> None:
    assert normalise_whitespace("  a \n\n b\t c ") == "a b c"




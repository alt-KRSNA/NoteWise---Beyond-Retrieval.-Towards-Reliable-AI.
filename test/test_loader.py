from app.ingestion.data_loader import load_pdf


def test_load_pdf():

    result = load_pdf("AI_Sample.pdf")

    assert result is not None
    assert result["file_name"] == "AI_Sample.pdf"
    assert isinstance(result["pages"], list)
    assert len(result["pages"]) > 0

    print("PDF loading test passed!")


if __name__ == "__main__":
    test_load_pdf()
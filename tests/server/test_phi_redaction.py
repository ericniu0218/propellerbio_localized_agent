from app.compliance.phi_redaction import redact_phi


def test_redacts_ssn():
    result = redact_phi("Patient SSN is 123-45-6789 on file.")
    assert "123-45-6789" not in result.text
    assert "[REDACTED:SSN]" in result.text
    assert result.redacted is True
    assert "SSN" in result.findings


def test_redacts_email_and_phone():
    result = redact_phi("Contact patient at jane.doe@example.com or (555) 123-4567.")
    assert "jane.doe@example.com" not in result.text
    assert "555" not in result.text or "[REDACTED:PHONE]" in result.text
    assert "EMAIL" in result.findings
    assert "PHONE" in result.findings


def test_redacts_mrn():
    result = redact_phi("MRN: 0012345 shows elevated ALT.")
    assert "0012345" not in result.text
    assert "MRN" in result.findings


def test_redacts_date_of_birth():
    result = redact_phi("DOB 04/12/1980 confirmed at intake.")
    assert "04/12/1980" not in result.text
    assert "DOB" in result.findings


def test_no_findings_on_clean_text():
    result = redact_phi("The compound showed a 15% reduction in tumor volume.")
    assert result.redacted is False
    assert result.findings == []
    assert result.text == "The compound showed a 15% reduction in tumor volume."


def test_multiple_findings_all_redacted_together():
    result = redact_phi("SSN 123-45-6789, email a@b.com, DOB 01/02/1990.")
    assert set(result.findings) == {"SSN", "EMAIL", "DOB"}
    assert "123-45-6789" not in result.text
    assert "a@b.com" not in result.text
    assert "01/02/1990" not in result.text

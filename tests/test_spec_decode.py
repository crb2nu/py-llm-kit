import unittest
from unittest.mock import MagicMock, patch

from llm_kit.spec_decode import spec_decode


class TestSpecDecode(unittest.IsolatedAsyncioTestCase):
    @patch("llm_kit.spec_decode.get_textgen_model")
    @patch("llm_kit.spec_decode.get_agent_model")
    async def test_immediate_approval(self, mock_get_agent, mock_get_textgen):
        mock_draft_llm = MagicMock()
        mock_draft_llm.invoke.return_value.content = "Draft content"
        mock_get_textgen.return_value = mock_draft_llm

        mock_verify_llm = MagicMock()
        mock_structured = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "APPROVED"
        mock_response.feedback = ""
        mock_structured.invoke.return_value = mock_response
        mock_verify_llm.with_structured_output.return_value = mock_structured
        mock_get_agent.return_value = mock_verify_llm

        result = await spec_decode("Test task")
        self.assertEqual(result, "Draft content")

    @patch("llm_kit.spec_decode.get_textgen_model")
    @patch("llm_kit.spec_decode.get_agent_model")
    async def test_revision_loop(self, mock_get_agent, mock_get_textgen):
        mock_draft_llm = MagicMock()
        mock_draft_llm.invoke.side_effect = [
            MagicMock(content="Bad draft"),
            MagicMock(content="Good draft"),
        ]
        mock_get_textgen.return_value = mock_draft_llm

        mock_verify_llm = MagicMock()
        mock_structured = MagicMock()

        response1 = MagicMock()
        response1.status = "REVISE"
        response1.feedback = "Fix it"

        response2 = MagicMock()
        response2.status = "APPROVED"
        response2.feedback = ""

        mock_structured.invoke.side_effect = [response1, response2]
        mock_verify_llm.with_structured_output.return_value = mock_structured
        mock_get_agent.return_value = mock_verify_llm

        result = await spec_decode("Test task")
        self.assertEqual(result, "Good draft")

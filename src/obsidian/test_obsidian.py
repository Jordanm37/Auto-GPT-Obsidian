"""Import the `os` module to access the environment variables from the `.env` file"""
from unittest.mock import mock_open, patch
import obsidian
import pytest
import os

"""Import the pytest package for unit tests"""


"""Import the partial from functools"""

"""Mock Unit Test Package for testing in mock states"""


class TestObsidianVault:
    """Testing Class for obsidian.py"""

    # OBSIDIAN_VAULT_NAME - the name of the obsidian vault
    # OBSIDIAN_VAULT_GIT_URL - the repository url (without .git) of the vault.
    # OBSIDIAN_FLASHCARD_SUBDIRECTORY - the subdirectory in which to create flashcards with spaced repition format/syntax.
    # OBSIDIAN_GITHUB_API_KEY - the API KEY to which responsibilities inside of the github repository ahave been allowed
    # OBSIDIAN_GITHUB_USERNAME - the username of the account housing the github repository and key.
    def simple_test(self):
        pass

    def test_arbitrary(self):
        """Test the arbitrary function"""
        assert 1 == 1

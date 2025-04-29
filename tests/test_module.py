"""Tests for module.py"""

from src.package.module import DummyClass


# Let's write some unit tests using pytest and arrange-act-assert pattern


def test_dummy_class():
    # Arrange
    dummy = DummyClass()

    # Act
    dummy.dummy_method()

    # Assert
    assert True

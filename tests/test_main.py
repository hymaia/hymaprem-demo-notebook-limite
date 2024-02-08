from src.main import hello_world


def test_hello_world():
    # Given
    expected = "Hello world!"

    # When
    actual = hello_world()

    # Then
    assert actual == expected

from utils.preprocessing import preprocess

def test_preprocess():
    text = "This is a test."
    expected_output = ["This", "test", "."]  